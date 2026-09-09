#include "SNNEngine.h"
#include "Constants.h"
#include "util/Log.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

namespace {

std::string openclBaseBuildOptions()
{
    return "-cl-fast-relaxed-math -cl-mad-enable";
}

[[nodiscard]] size_t spikedOnlyGlobalWorkSize(std::uint32_t spiked_only_count)
{
    constexpr size_t k_local_work_size = 32u;
    if (spiked_only_count == 0u) {
        return 0u;
    }
    return static_cast<size_t>(spiked_only_count) * k_local_work_size;
}

}  // namespace

SNNEngine::SNNEngine() = default;

SNNEngine::~SNNEngine()
{
    freeDeviceBuffers();
    releaseOpenCLKernelsAndPrograms();
}

void SNNEngine::allocateBuffers(
    cl_device_id       device,
    cl_context         context,
    cl_command_queue   queue,
    const HostOrganismBuffers& organism_buffers)
{
    device_ = device;
    queue_ = queue;
    context_ = context;
    h_organism_buffers_ = organism_buffers;

    // Copy host-side C++ structs first, then allocate OpenCL mirrors from those values.
    const auto& g = h_organism_buffers_.h_global_params_;
    const auto& n = h_organism_buffers_.h_neuron_buffers_;
    const auto& c = h_organism_buffers_.h_csr_csc_buffers_;
    const auto& p = h_organism_buffers_.h_pipeline_buffers_;


    //global_
    d_global_.d_current_tick = g.h_current_tick;
    d_global_.d_words_per_row = g.h_words_per_row;
    d_global_.d_num_of_input_neurons = g.h_num_of_input_neurons;
    d_global_.d_num_output_slots = g.h_num_output_slots;
    d_global_.d_feedback_temporal_decay_index =
        static_cast<cl_uint>(kernel_input_layout::k_temporal_decay_input);
    d_global_.d_last_click_distance_index =
        static_cast<cl_uint>(kernel_input_layout::k_last_click_distance_input);
    d_global_.d_trace_tau = g.h_trace_tau;
    d_global_.d_num_neurons = g.h_num_neurons;
    d_global_.d_num_synapses = g.h_num_synapses;
    d_global_.d_max_trace_len = g.h_max_trace_len;
    d_global_.d_max_delay = g.h_max_delay;
    d_global_.d_click_distance_weight = g.h_click_distance_weight;
    d_global_.d_homeo_weight_mod = g.h_homeo_weight_mod;
    d_global_.d_homeo_pot_mod = g.h_homeo_pot_mod;
    d_global_.d_homeo_dep_mod = g.h_homeo_dep_mod;
    d_global_.d_homeo_thresh_mod = g.h_homeo_thresh_mod;
    d_global_.d_homeo_rate_max_distance = g.h_homeo_rate_max_distance;
    d_global_.d_ticks_between_homeostasis = g.h_ticks_between_homeostasis;
    d_global_.d_fire_rate_tau = std::max(1.0f, static_cast<cl_float>(g.h_ticks_between_homeostasis) * 1.5f);
    d_global_.d_capacitance_C = g.h_capacitance_C;
    d_global_.d_leak_g = g.h_leak_g;
    d_global_.d_leak_E = g.h_leak_E;
    d_global_.d_slope_delta_T = g.h_slope_delta_T;
    d_global_.d_v_peak = g.h_v_peak;
    d_global_.d_dt_ms = g.h_dt_ms;
    cl_uint d_inhibitory_spike_rate = g.h_inhibitory_spike_rate;
    cl_uint d_excitatory_spike_rate = g.h_excitatory_spike_rate;
    std::vector<cl_uint> d_output_slot_neuron_counts = g.h_output_slot_neuron_counts;

    //neuron_buffers
    std::vector<cl_float> d_V = n.h_V;
    std::vector<cl_float> d_W = n.h_W;
    std::vector<cl_float> d_threshold_V = n.h_threshold_V;
    std::vector<cl_uint> d_refract_end = n.h_refract_end;
    std::vector<cl_uint> d_refractory_length = n.h_refractory_length;
    std::vector<cl_uint> d_last_delievery_update = n.h_last_delievery_update;
    std::vector<cl_uint> d_last_spike = n.h_last_spike;
    std::vector<cl_float> d_spike_rate = n.h_spike_rate;
    std::vector<cl_float> d_spike_rate_target = n.h_spike_rate_target;
    std::vector<cl_float> d_a = n.h_a;
    std::vector<cl_float> d_b = n.h_b;
    std::vector<cl_float> d_tau_w = n.h_tau_w;
    std::vector<cl_float> d_v_rest = n.h_v_rest;
    std::vector<cl_uint> d_neuron_kind = n.h_neuron_kind;
    std::vector<cl_uint> d_neuron_kind_target = n.h_neuron_kind_target;
    //csr_csc_buffers
    std::vector<cl_uint> d_csr_row_ptr = c.h_csr_row_ptr;
    std::vector<cl_uint> d_csr_col_idx = c.h_csr_col_idx;
    std::vector<cl_uint> d_csc_row_ptr = c.h_csc_row_ptr;
    std::vector<cl_uint> d_csc_col_idx = c.h_csc_col_idx;
    std::vector<cl_uint> d_csc_to_csr_index_mapping = c.h_csc_to_csr_index_mapping;
    std::vector<cl_float> d_weight = c.h_weight;
    std::vector<cl_float> d_delay = c.h_delay;
    std::vector<cl_float> d_potentiation_bias = c.h_potentiation_bias;
    std::vector<cl_float> d_depression_bias = c.h_depression_bias;
    //pipeline_buffers
    std::vector<cl_float> d_delayed_charge = p.h_delayed_charge;
    std::vector<cl_uint> d_dirty_delayed_charge = p.h_dirty_delayed_charge;
    std::vector<cl_uchar> d_spiked = p.h_spiked;
    std::vector<cl_uint> d_spiked_only = p.h_spiked_only;
    cl_uint d_spiked_only_offset = p.h_spiked_only_offset;
    std::vector<cl_float> d_input = p.h_input;
    std::vector<cl_uint> d_output = p.h_output;

    cl_int err = CL_SUCCESS;

    d_global_.d_inhibitory_spike_rate = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint),
        &d_inhibitory_spike_rate,
        &err);
    d_global_.d_excitatory_spike_rate = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint),
        &d_excitatory_spike_rate,
        &err);
    d_global_.d_output_slot_neuron_counts = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_output_slot_neuron_counts.size() * sizeof(cl_uint), d_output_slot_neuron_counts.data(), &err);

    //neurons_
    d_neurons_.d_V = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_V.size() * sizeof(cl_float), d_V.data(), &err);
    d_neurons_.d_W = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_W.size() * sizeof(cl_float), d_W.data(), &err);
    d_neurons_.d_threshold_V = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_threshold_V.size() * sizeof(cl_float), d_threshold_V.data(), &err);
    d_neurons_.d_refract_end = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_refract_end.size() * sizeof(cl_uint), d_refract_end.data(), &err);
    d_neurons_.d_refractory_length = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_refractory_length.size() * sizeof(cl_uint), d_refractory_length.data(), &err);
    d_neurons_.d_last_delievery_update = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_last_delievery_update.size() * sizeof(cl_uint), d_last_delievery_update.data(), &err);
    d_neurons_.d_last_spike = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_last_spike.size() * sizeof(cl_uint), d_last_spike.data(), &err);
    d_neurons_.d_spike_rate = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_spike_rate.size() * sizeof(cl_float), d_spike_rate.data(), &err);
    d_neurons_.d_spike_rate_target = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_spike_rate_target.size() * sizeof(cl_float), d_spike_rate_target.data(), &err);
    d_neurons_.d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_a.size() * sizeof(cl_float), d_a.data(), &err);
    d_neurons_.d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_b.size() * sizeof(cl_float), d_b.data(), &err);
    d_neurons_.d_tau_w = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_tau_w.size() * sizeof(cl_float), d_tau_w.data(), &err);
    d_neurons_.d_v_rest = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_v_rest.size() * sizeof(cl_float), d_v_rest.data(), &err);
    d_neurons_.d_neuron_kind = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_neuron_kind.size() * sizeof(cl_uint), d_neuron_kind.data(), &err);
    d_neurons_.d_neuron_kind_target = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_neuron_kind_target.size() * sizeof(cl_uint), d_neuron_kind_target.data(), &err);

    //csr_csc_buffers
    d_csr_csc_.d_csr_row_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_csr_row_ptr.size() * sizeof(cl_uint), d_csr_row_ptr.data(), &err);
    d_csr_csc_.d_csr_col_idx = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_csr_col_idx.size() * sizeof(cl_uint), d_csr_col_idx.data(), &err);
    d_csr_csc_.d_csc_row_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_csc_row_ptr.size() * sizeof(cl_uint), d_csc_row_ptr.data(), &err);
    d_csr_csc_.d_csc_col_idx = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_csc_col_idx.size() * sizeof(cl_uint), d_csc_col_idx.data(), &err);
    d_csr_csc_.d_csc_to_csr_index_mapping = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_csc_to_csr_index_mapping.size() * sizeof(cl_uint), d_csc_to_csr_index_mapping.data(), &err);
    d_csr_csc_.d_weight = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_weight.size() * sizeof(cl_float), d_weight.data(), &err);
    d_csr_csc_.d_delay = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d_delay.size() * sizeof(cl_float), d_delay.data(), &err);
    d_csr_csc_.d_potentiation_bias = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_potentiation_bias.size() * sizeof(cl_float), d_potentiation_bias.data(), &err);
    d_csr_csc_.d_depression_bias = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_depression_bias.size() * sizeof(cl_float), d_depression_bias.data(), &err);

    //pipeline_buffers
    d_pipeline_.d_delayed_charge = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_delayed_charge.size() * sizeof(cl_float), d_delayed_charge.data(), &err);
    d_pipeline_.d_dirty_delayed_charge = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_dirty_delayed_charge.size() * sizeof(cl_uint), d_dirty_delayed_charge.data(), &err);
    d_pipeline_.d_spiked = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_spiked.size() * sizeof(cl_uchar), d_spiked.data(), &err);
    d_pipeline_.d_spiked_only = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_spiked_only.size() * sizeof(cl_uint), d_spiked_only.data(), &err);
    d_pipeline_.d_spiked_only_offset = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &d_spiked_only_offset, &err);
    d_pipeline_.d_trace_lut = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p.h_trace_lut.size() * sizeof(cl_float), const_cast<cl_float*>(p.h_trace_lut.data()), &err);
    d_pipeline_.d_input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_input.size() * sizeof(cl_float), d_input.data(), &err);
    d_pipeline_.d_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, d_output.size() * sizeof(cl_uint), d_output.data(), &err);
}

void SNNEngine::initializeDelieveryLeakFireKernel()
{
    if (delievery_leak_fire_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/delievery_leak_fire.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/delievery_leak_fire.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open delievery_leak_fire.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    delievery_leak_fire_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !delievery_leak_fire_program_) {
        Util::Errors::stop("Failed to create delivery/leak/fire program");
    }
    err = clBuildProgram(
        delievery_leak_fire_program_,
        1,
        &device_,
        "-cl-fast-relaxed-math -cl-mad-enable",
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(
            delievery_leak_fire_program_,
            device_,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(
            delievery_leak_fire_program_,
            device_,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            log.data(),
            nullptr);
        Util::Errors::stop("delivery_decay_fire build error:\n" + log);
    }
    delievery_leak_fire_kernel_ = clCreateKernel(delievery_leak_fire_program_, "delivery_decay_fire", &err);
    if (err != CL_SUCCESS || !delievery_leak_fire_kernel_) {
        Util::Errors::stop("Failed to create delivery_decay_fire kernel");
    }
}

void SNNEngine::initializeInputToSpikedKernel()
{
    if (input_to_spiked_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/input_to_spiked.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/input_to_spiked.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open input_to_spiked.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    input_to_spiked_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !input_to_spiked_program_) {
        Util::Errors::stop("Failed to create input_to_spiked program");
    }
    err = clBuildProgram(
        input_to_spiked_program_,
        1,
        &device_,
        "-cl-fast-relaxed-math -cl-mad-enable",
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(input_to_spiked_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(input_to_spiked_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        Util::Errors::stop("input_to_spiked build error:\n" + log);
    }
    input_to_spiked_kernel_ = clCreateKernel(input_to_spiked_program_, "input_to_spiked", &err);
    if (err != CL_SUCCESS || !input_to_spiked_kernel_) {
        Util::Errors::stop("Failed to create input_to_spiked kernel");
    }
}

void SNNEngine::initializeCompactionKernel()
{
    if (compaction_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/spiked_only_compaction.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/spiked_only_compaction.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open spiked_only_compaction.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    compaction_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !compaction_program_) {
        Util::Errors::stop("Failed to create spiked_only_compaction program");
    }
    err = clBuildProgram(
        compaction_program_,
        1,
        &device_,
        "-cl-fast-relaxed-math -cl-mad-enable",
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(compaction_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(compaction_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        Util::Errors::stop("spiked_only_compaction build error:\n" + log);
    }
    compaction_kernel_ = clCreateKernel(compaction_program_, "spiked_only_compaction", &err);
    if (err != CL_SUCCESS || !compaction_kernel_) {
        Util::Errors::stop("Failed to create spiked_only_compaction kernel");
    }
}

void SNNEngine::initializeSTDPKernel()
{
    if (stdp_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/stdp.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/stdp.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open stdp.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    stdp_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !stdp_program_) {
        Util::Errors::stop("Failed to create stdp program");
    }
    const std::string build_options = openclBaseBuildOptions();
    err = clBuildProgram(
        stdp_program_,
        1,
        &device_,
        build_options.c_str(),
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(stdp_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(stdp_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        Util::Errors::stop("stdp build error:\n" + log);
    }
    stdp_kernel_ = clCreateKernel(stdp_program_, "stdp", &err);
    if (err != CL_SUCCESS || !stdp_kernel_) {
        Util::Errors::stop("Failed to create stdp kernel");
    }
}

void SNNEngine::initializePropagationKernel()
{
    if (propagation_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/propagation.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/propagation.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open propagation.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    propagation_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !propagation_program_) {
        Util::Errors::stop("Failed to create propagation program");
    }
    const std::string build_options = openclBaseBuildOptions();
    err = clBuildProgram(
        propagation_program_,
        1,
        &device_,
        build_options.c_str(),
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(propagation_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(propagation_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        Util::Errors::stop("propagation build error:\n" + log);
    }
    propagation_kernel_ = clCreateKernel(propagation_program_, "propagation", &err);
    if (err != CL_SUCCESS || !propagation_kernel_) {
        Util::Errors::stop("Failed to create propagation kernel");
    }
}

void SNNEngine::initializeHomeostasisKernel()
{
    if (homeostasis_kernel_) {
        return;
    }

    std::ifstream file("app/kernels/homeostasis.cl");
    if (!file.is_open()) {
        file.open("../app/kernels/homeostasis.cl");
    }
    if (!file.is_open()) {
        Util::Errors::stop("Failed to open homeostasis.cl from app/kernels");
    }
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.length();

    cl_int err = CL_SUCCESS;
    homeostasis_program_ = clCreateProgramWithSource(context_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !homeostasis_program_) {
        Util::Errors::stop("Failed to create homeostasis program");
    }
    err = clBuildProgram(
        homeostasis_program_,
        1,
        &device_,
        "-cl-fast-relaxed-math -cl-mad-enable",
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(homeostasis_program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(homeostasis_program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        Util::Errors::stop("homeostasis build error:\n" + log);
    }
    homeostasis_kernel_ = clCreateKernel(homeostasis_program_, "homeostasis", &err);
    if (err != CL_SUCCESS || !homeostasis_kernel_) {
        Util::Errors::stop("Failed to create homeostasis kernel");
    }
}

bool SNNEngine::clearOutputBuffer(){
    if (!queue_ || !d_pipeline_.d_output) {
        return false;
    }
    const cl_uint zero = 0u;
    cl_int err = clEnqueueFillBuffer(
        queue_,
        d_pipeline_.d_output,
        &zero,
        sizeof(zero),
        0,
        sizeof(cl_uint) * d_global_.d_num_output_slots,
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("clearOutputBuffer: clEnqueueFillBuffer failed, err=" + std::to_string(err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::clearPropagationSpikeRates()
{
    if (!queue_ || !d_global_.d_inhibitory_spike_rate || !d_global_.d_excitatory_spike_rate) {
        return false;
    }
    const cl_uint zero = 0u;
    cl_int err = clEnqueueFillBuffer(
        queue_,
        d_global_.d_inhibitory_spike_rate,
        &zero,
        sizeof(zero),
        0,
        sizeof(cl_uint),
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("clearPropagationSpikeRates: failed to clear inhibitory_spike_rate, err=" + std::to_string(err) + '\n');
        return false;
    }
    err = clEnqueueFillBuffer(
        queue_,
        d_global_.d_excitatory_spike_rate,
        &zero,
        sizeof(zero),
        0,
        sizeof(cl_uint),
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("clearPropagationSpikeRates: failed to clear excitatory_spike_rate, err=" + std::to_string(err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::inputToSpiked()
{
    if (!queue_ || !d_pipeline_.d_input || !d_pipeline_.d_spiked) {
        Util::Bugs::write("inputToSpiked: missing OpenCL queue, d_input, or d_spiked\n");
        return false;
    }
    if (!input_to_spiked_kernel_) {
        Util::Bugs::write("inputToSpiked: input_to_spiked_kernel_ is null\n");
        return false;
    }

    if (d_global_.d_num_neurons == 0u) {
        Util::Bugs::write("inputToSpiked: organism has no neurons\n");
        return false;
    }

    const cl_uint current_tick = static_cast<cl_uint>(d_global_.d_current_tick);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(input_to_spiked_kernel_, 0, sizeof(cl_uint), &current_tick);
    err |= clSetKernelArg(input_to_spiked_kernel_, 1, sizeof(cl_mem), &d_pipeline_.d_input);
    err |= clSetKernelArg(input_to_spiked_kernel_, 2, sizeof(cl_mem), &d_pipeline_.d_spiked);
    err |= clSetKernelArg(input_to_spiked_kernel_, 3, sizeof(cl_mem), &d_neurons_.d_last_spike);
    err |= clSetKernelArg(input_to_spiked_kernel_, 4, sizeof(cl_mem), &d_neurons_.d_neuron_kind);
    err |= clSetKernelArg(input_to_spiked_kernel_, 5, sizeof(cl_mem), &d_neurons_.d_neuron_kind_target);
    err |= clSetKernelArg(input_to_spiked_kernel_, 6, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(input_to_spiked_kernel_, 7, sizeof(cl_uint), &d_global_.d_num_of_input_neurons);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("inputToSpiked: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const size_t global_work_size[1] = {static_cast<size_t>(d_global_.d_num_neurons)};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        input_to_spiked_kernel_,
        1,
        nullptr,
        global_work_size,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("inputToSpiked: clEnqueueNDRangeKernel failed, err=" + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

void SNNEngine::uploadToInputBuffer(size_t offset, const std::vector<float>& h_input_buffer)
{
    if (!queue_ || !d_pipeline_.d_input) {
        Util::Bugs::write("uploadToInputBuffer: missing OpenCL queue or d_input\n");
        return;
    }
    const std::size_t total_floats =
        static_cast<std::size_t>(d_global_.d_num_of_input_neurons);
    if (offset > total_floats || h_input_buffer.size() > total_floats - offset) {
        Util::Bugs::write(
            "uploadToInputBuffer: range out of bounds offset="
            + std::to_string(offset)
            + " length="
            + std::to_string(h_input_buffer.size())
            + " total="
            + std::to_string(total_floats)
            + '\n');
        return;
    }
    const cl_int err = clEnqueueWriteBuffer(
        queue_,
        d_pipeline_.d_input,
        CL_TRUE,
        offset * sizeof(cl_float),
        h_input_buffer.size() * sizeof(cl_float),
        h_input_buffer.data(),
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadToInputBuffer: clEnqueueWriteBuffer failed, err=" + std::to_string(err) + '\n');
        return;
    }
}

bool SNNEngine::uploadDelieveryLeakFireKernel()
{
    if (!queue_) {
        Util::Bugs::write("uploadDelieveryLeakFireKernel: missing OpenCL queue\n");
        return false;
    }
    if (!delievery_leak_fire_kernel_) {
        Util::Bugs::write("uploadDelieveryLeakFireKernel: delievery_leak_fire_kernel_ is null\n");
        return false;
    }

    const cl_uint current_tick = static_cast<cl_uint>(d_global_.d_current_tick);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 0, sizeof(cl_uint), &current_tick);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 1, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 2, sizeof(cl_uint), &d_global_.d_words_per_row);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 3, sizeof(cl_uint), &d_global_.d_max_delay);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 4, sizeof(cl_float), &d_global_.d_dt_ms);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 5, sizeof(cl_float), &d_global_.d_capacitance_C);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 6, sizeof(cl_float), &d_global_.d_leak_g);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 7, sizeof(cl_float), &d_global_.d_leak_E);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 8, sizeof(cl_float), &d_global_.d_slope_delta_T);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 9, sizeof(cl_float), &d_global_.d_v_peak);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 10, sizeof(cl_mem), &d_neurons_.d_V);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 11, sizeof(cl_mem), &d_neurons_.d_W);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 12, sizeof(cl_mem), &d_neurons_.d_last_delievery_update);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 13, sizeof(cl_mem), &d_neurons_.d_refract_end);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 14, sizeof(cl_mem), &d_neurons_.d_refractory_length);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 15, sizeof(cl_mem), &d_neurons_.d_a);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 16, sizeof(cl_mem), &d_neurons_.d_b);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 17, sizeof(cl_mem), &d_neurons_.d_tau_w);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 18, sizeof(cl_mem), &d_neurons_.d_threshold_V);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 19, sizeof(cl_mem), &d_neurons_.d_v_rest);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 20, sizeof(cl_mem), &d_neurons_.d_spike_rate);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 21, sizeof(cl_mem), &d_pipeline_.d_dirty_delayed_charge);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 22, sizeof(cl_mem), &d_pipeline_.d_delayed_charge);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 23, sizeof(cl_mem), &d_pipeline_.d_spiked);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 24, sizeof(cl_mem), &d_neurons_.d_last_spike);
    err |= clSetKernelArg(delievery_leak_fire_kernel_, 25, sizeof(cl_float), &d_global_.d_fire_rate_tau);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadDelieveryLeakFireKernel: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const size_t global_work_size[1] = {
        static_cast<size_t>(d_global_.d_num_neurons)};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        delievery_leak_fire_kernel_,
        1,
        nullptr,
        global_work_size,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("uploadDelieveryLeakFireKernel: clEnqueueNDRangeKernel failed, err="
            + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::uploadCompactionKernel()
{
    if (!queue_ || !compaction_kernel_) {
        Util::Bugs::write("uploadCompactionKernel: missing OpenCL queue or compaction_kernel_\n");
        return false;
    }
    if (d_global_.d_num_neurons == 0u) {
        Util::Bugs::write("uploadCompactionKernel: organism has no neurons\n");
        return false;
    }

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(compaction_kernel_, 0, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(compaction_kernel_, 1, sizeof(cl_mem), &d_pipeline_.d_spiked);
    err |= clSetKernelArg(compaction_kernel_, 2, sizeof(cl_mem), &d_pipeline_.d_spiked_only);
    err |= clSetKernelArg(compaction_kernel_, 3, sizeof(cl_mem), &d_pipeline_.d_spiked_only_offset);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadCompactionKernel: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const cl_uint zero = 0u;
    const cl_int clear_err = clEnqueueWriteBuffer(
        queue_,
        d_pipeline_.d_spiked_only_offset,
        CL_TRUE,
        0,
        sizeof(zero),
        &zero,
        0,
        nullptr,
        nullptr);
    if (clear_err != CL_SUCCESS) {
        Util::Bugs::write("uploadCompactionKernel: reset spiked_only offset failed, err=" + std::to_string(clear_err) + '\n');
        return false;
    }

    const size_t local_work_size[1] = {32u};
    const size_t remainder = d_global_.d_num_neurons % local_work_size[0];
    const size_t global_work_size[1] = {
        remainder == 0u ? d_global_.d_num_neurons : d_global_.d_num_neurons + local_work_size[0] - remainder};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        compaction_kernel_,
        1,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("uploadCompactionKernel: clEnqueueNDRangeKernel failed, err="
            + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::uploadSTDPKernel(std::uint32_t spiked_only_count)
{
    if (!queue_ || !stdp_kernel_) {
        Util::Bugs::write("uploadSTDPKernel: missing OpenCL queue or stdp_kernel_\n");
        return false;
    }
    if (d_global_.d_num_neurons == 0u) {
        Util::Bugs::write("uploadSTDPKernel: organism has no neurons\n");
        return false;
    }
    if (spiked_only_count == 0u) {
        return true;
    }

    const cl_uint current_tick = static_cast<cl_uint>(d_global_.d_current_tick);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(stdp_kernel_, 0, sizeof(cl_uint), &current_tick);
    err |= clSetKernelArg(stdp_kernel_, 1, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(stdp_kernel_, 2, sizeof(cl_uint), &d_global_.d_max_trace_len);
    err |= clSetKernelArg(stdp_kernel_, 3, sizeof(cl_float), &d_global_.d_click_distance_weight);
    err |= clSetKernelArg(stdp_kernel_, 4, sizeof(cl_uint), &d_global_.d_feedback_temporal_decay_index);
    err |= clSetKernelArg(stdp_kernel_, 5, sizeof(cl_uint), &d_global_.d_last_click_distance_index);
    err |= clSetKernelArg(stdp_kernel_, 6, sizeof(cl_mem), &d_csr_csc_.d_csc_row_ptr);
    err |= clSetKernelArg(stdp_kernel_, 7, sizeof(cl_mem), &d_csr_csc_.d_csc_col_idx);
    err |= clSetKernelArg(stdp_kernel_, 8, sizeof(cl_mem), &d_csr_csc_.d_csc_to_csr_index_mapping);
    err |= clSetKernelArg(stdp_kernel_, 9, sizeof(cl_mem), &d_csr_csc_.d_weight);
    err |= clSetKernelArg(stdp_kernel_, 10, sizeof(cl_mem), &d_csr_csc_.d_delay);
    err |= clSetKernelArg(stdp_kernel_, 11, sizeof(cl_mem), &d_csr_csc_.d_potentiation_bias);
    err |= clSetKernelArg(stdp_kernel_, 12, sizeof(cl_mem), &d_csr_csc_.d_depression_bias);
    err |= clSetKernelArg(stdp_kernel_, 13, sizeof(cl_mem), &d_pipeline_.d_input);
    err |= clSetKernelArg(stdp_kernel_, 14, sizeof(cl_mem), &d_pipeline_.d_trace_lut);
    err |= clSetKernelArg(stdp_kernel_, 15, sizeof(cl_mem), &d_pipeline_.d_spiked_only);
    err |= clSetKernelArg(stdp_kernel_, 16, sizeof(cl_mem), &d_pipeline_.d_spiked_only_offset);
    err |= clSetKernelArg(stdp_kernel_, 17, sizeof(cl_mem), &d_neurons_.d_last_spike);
    err |= clSetKernelArg(stdp_kernel_, 18, sizeof(cl_uint), &d_global_.d_num_synapses);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadSTDPKernel: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const size_t local_work_size[1] = {32u};
    const size_t global_work_size[1] = {spikedOnlyGlobalWorkSize(spiked_only_count)};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        stdp_kernel_,
        1,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("uploadSTDPKernel: clEnqueueNDRangeKernel failed, err="
            + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::uploadPropagationKernel(std::uint32_t spiked_only_count)
{
    if (!queue_ || !propagation_kernel_) {
        Util::Bugs::write("uploadPropagationKernel: missing OpenCL queue or propagation_kernel_\n");
        return false;
    }
    if (d_global_.d_num_neurons == 0u) {
        Util::Bugs::write("uploadPropagationKernel: organism has no neurons\n");
        return false;
    }
    if (spiked_only_count == 0u) {
        return true;
    }

    const cl_uint current_tick = static_cast<cl_uint>(d_global_.d_current_tick);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(propagation_kernel_, 0, sizeof(cl_uint), &current_tick);
    err |= clSetKernelArg(propagation_kernel_, 1, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(propagation_kernel_, 2, sizeof(cl_uint), &d_global_.d_words_per_row);
    err |= clSetKernelArg(propagation_kernel_, 3, sizeof(cl_uint), &d_global_.d_max_delay);
    err |= clSetKernelArg(propagation_kernel_, 4, sizeof(cl_uint), &d_global_.d_num_of_input_neurons);
    err |= clSetKernelArg(propagation_kernel_, 5, sizeof(cl_uint), &d_global_.d_num_output_slots);
    err |= clSetKernelArg(propagation_kernel_, 6, sizeof(cl_mem), &d_csr_csc_.d_csr_row_ptr);
    err |= clSetKernelArg(propagation_kernel_, 7, sizeof(cl_mem), &d_csr_csc_.d_csr_col_idx);
    err |= clSetKernelArg(propagation_kernel_, 8, sizeof(cl_mem), &d_csr_csc_.d_weight);
    err |= clSetKernelArg(propagation_kernel_, 9, sizeof(cl_mem), &d_csr_csc_.d_delay);
    err |= clSetKernelArg(propagation_kernel_, 10, sizeof(cl_mem), &d_pipeline_.d_spiked_only);
    err |= clSetKernelArg(propagation_kernel_, 11, sizeof(cl_mem), &d_pipeline_.d_spiked_only_offset);
    err |= clSetKernelArg(propagation_kernel_, 12, sizeof(cl_mem), &d_pipeline_.d_delayed_charge);
    err |= clSetKernelArg(propagation_kernel_, 13, sizeof(cl_mem), &d_pipeline_.d_dirty_delayed_charge);
    err |= clSetKernelArg(propagation_kernel_, 14, sizeof(cl_mem), &d_pipeline_.d_input);
    err |= clSetKernelArg(propagation_kernel_, 15, sizeof(cl_mem), &d_pipeline_.d_output);
    err |= clSetKernelArg(propagation_kernel_, 16, sizeof(cl_mem), &d_neurons_.d_neuron_kind);
    err |= clSetKernelArg(propagation_kernel_, 17, sizeof(cl_mem), &d_neurons_.d_neuron_kind_target);
    err |= clSetKernelArg(propagation_kernel_, 18, sizeof(cl_mem), &d_global_.d_inhibitory_spike_rate);
    err |= clSetKernelArg(propagation_kernel_, 19, sizeof(cl_mem), &d_global_.d_excitatory_spike_rate);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadPropagationKernel: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const size_t local_work_size[1] = {32u};
    const size_t global_work_size[1] = {spikedOnlyGlobalWorkSize(spiked_only_count)};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        propagation_kernel_,
        1,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("uploadPropagationKernel: clEnqueueNDRangeKernel failed, err="
            + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::uploadHomeostasisKernel()
{
    if (!queue_ || !homeostasis_kernel_) {
        Util::Bugs::write("uploadHomeostasisKernel: missing OpenCL queue or homeostasis_kernel_\n");
        return false;
    }
    if (d_global_.d_num_neurons == 0u) {
        Util::Bugs::write("uploadHomeostasisKernel: organism has no neurons\n");
        return false;
    }

    const cl_uint current_tick = static_cast<cl_uint>(d_global_.d_current_tick);
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(homeostasis_kernel_, 0, sizeof(cl_uint), &current_tick);
    err |= clSetKernelArg(homeostasis_kernel_, 1, sizeof(cl_uint), &d_global_.d_num_neurons);
    err |= clSetKernelArg(homeostasis_kernel_, 2, sizeof(cl_uint), &d_global_.d_ticks_between_homeostasis);
    err |= clSetKernelArg(homeostasis_kernel_, 3, sizeof(cl_float), &d_global_.d_homeo_weight_mod);
    err |= clSetKernelArg(homeostasis_kernel_, 4, sizeof(cl_float), &d_global_.d_homeo_pot_mod);
    err |= clSetKernelArg(homeostasis_kernel_, 5, sizeof(cl_float), &d_global_.d_homeo_dep_mod);
    err |= clSetKernelArg(homeostasis_kernel_, 6, sizeof(cl_float), &d_global_.d_homeo_thresh_mod);
    err |= clSetKernelArg(homeostasis_kernel_, 7, sizeof(cl_float), &d_global_.d_homeo_rate_max_distance);
    err |= clSetKernelArg(homeostasis_kernel_, 8, sizeof(cl_mem), &d_csr_csc_.d_csc_row_ptr);
    err |= clSetKernelArg(homeostasis_kernel_, 9, sizeof(cl_mem), &d_csr_csc_.d_csc_to_csr_index_mapping);
    err |= clSetKernelArg(homeostasis_kernel_, 10, sizeof(cl_mem), &d_csr_csc_.d_weight);
    err |= clSetKernelArg(homeostasis_kernel_, 11, sizeof(cl_mem), &d_csr_csc_.d_potentiation_bias);
    err |= clSetKernelArg(homeostasis_kernel_, 12, sizeof(cl_mem), &d_csr_csc_.d_depression_bias);
    err |= clSetKernelArg(homeostasis_kernel_, 13, sizeof(cl_mem), &d_neurons_.d_spike_rate);
    err |= clSetKernelArg(homeostasis_kernel_, 14, sizeof(cl_mem), &d_neurons_.d_spike_rate_target);
    err |= clSetKernelArg(homeostasis_kernel_, 15, sizeof(cl_mem), &d_neurons_.d_threshold_V);
    err |= clSetKernelArg(homeostasis_kernel_, 16, sizeof(cl_uint), &d_global_.d_num_synapses);
    err |= clSetKernelArg(homeostasis_kernel_, 17, sizeof(cl_float), &d_global_.d_fire_rate_tau);
    err |= clSetKernelArg(homeostasis_kernel_, 18, sizeof(cl_mem), &d_neurons_.d_last_delievery_update);
    if (err != CL_SUCCESS) {
        Util::Bugs::write("uploadHomeostasisKernel: clSetKernelArg failed, err=" + std::to_string(err) + '\n');
        return false;
    }

    const size_t local_work_size[1] = {32u};
    const size_t global_work_size[1] = {
        static_cast<size_t>(d_global_.d_num_neurons) * local_work_size[0]};
    const cl_int enqueue_err = clEnqueueNDRangeKernel(
        queue_,
        homeostasis_kernel_,
        1,
        nullptr,
        global_work_size,
        local_work_size,
        0,
        nullptr,
        nullptr);
    if (enqueue_err != CL_SUCCESS) {
        Util::Bugs::write("uploadHomeostasisKernel: clEnqueueNDRangeKernel failed, err="
            + std::to_string(enqueue_err) + '\n');
        return false;
    }
    return true;
}

bool SNNEngine::readSpikedOnlyTotalCount(std::uint32_t& out_count)
{
    if (!queue_ || !d_pipeline_.d_spiked_only_offset) {
        Util::Bugs::write("readSpikedOnlyTotalCount: missing queue or spiked_only_offset buffer\n");
        return false;
    }
    clFinish(queue_);
    cl_uint v = 0u;
    const cl_int err = clEnqueueReadBuffer(
        queue_,
        d_pipeline_.d_spiked_only_offset,
        CL_TRUE,
        0,
        sizeof(v),
        &v,
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write(
            "readSpikedOnlyTotalCount: read failed, err=" + std::to_string(err) + '\n');
        return false;
    }
    out_count = static_cast<std::uint32_t>(v);
    return true;
}

bool SNNEngine::readPropagationSpikeRates(
    std::uint32_t& out_inhibitory,
    std::uint32_t& out_excitatory)
{
    if (!queue_ || !d_global_.d_inhibitory_spike_rate || !d_global_.d_excitatory_spike_rate) {
        Util::Bugs::write("readPropagationSpikeRates: missing queue or spike-rate buffers\n");
        return false;
    }
    clFinish(queue_);
    cl_uint inhibitory = 0u;
    cl_uint excitatory = 0u;
    cl_int err = clEnqueueReadBuffer(
        queue_,
        d_global_.d_inhibitory_spike_rate,
        CL_TRUE,
        0,
        sizeof(inhibitory),
        &inhibitory,
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write(
            "readPropagationSpikeRates: inhibitory read failed, err=" + std::to_string(err) + '\n');
        return false;
    }
    err = clEnqueueReadBuffer(
        queue_,
        d_global_.d_excitatory_spike_rate,
        CL_TRUE,
        0,
        sizeof(excitatory),
        &excitatory,
        0,
        nullptr,
        nullptr);
    if (err != CL_SUCCESS) {
        Util::Bugs::write(
            "readPropagationSpikeRates: excitatory read failed, err=" + std::to_string(err) + '\n');
        return false;
    }
    out_inhibitory = static_cast<std::uint32_t>(inhibitory);
    out_excitatory = static_cast<std::uint32_t>(excitatory);
    return true;
}

void SNNEngine::getOutputBuffer(std::vector<uint32_t>& h_output){
    if (!queue_ || !d_pipeline_.d_output) {
        Util::Bugs::write("getOutputBuffer: missing OpenCL queue or d_output\n");
        return;
    }
    if (h_output.size() < d_global_.d_num_output_slots) {
        Util::Errors::stop(
            "getOutputBuffer: h_output.size() ("
            + std::to_string(h_output.size())
            + ") < d_num_output_slots ("
            + std::to_string(d_global_.d_num_output_slots)
            + ')');
    }

    clEnqueueReadBuffer(queue_, d_pipeline_.d_output, CL_TRUE, 0, sizeof(cl_uint) * d_global_.d_num_output_slots, h_output.data(), 0, nullptr, nullptr);
}

void SNNEngine::freeDeviceBuffers()
{
    if (!queue_) {
        return;
    }
    clFinish(queue_);

    auto release_mem = [](cl_mem& mem) {
        if (mem) {
            clReleaseMemObject(mem);
            mem = nullptr;
        }
    };

    release_mem(d_global_.d_inhibitory_spike_rate);
    release_mem(d_global_.d_excitatory_spike_rate);
    release_mem(d_global_.d_output_slot_neuron_counts);

    release_mem(d_neurons_.d_V);
    release_mem(d_neurons_.d_W);
    release_mem(d_neurons_.d_threshold_V);
    release_mem(d_neurons_.d_refract_end);
    release_mem(d_neurons_.d_refractory_length);
    release_mem(d_neurons_.d_last_delievery_update);
    release_mem(d_neurons_.d_last_spike);
    release_mem(d_neurons_.d_spike_rate);
    release_mem(d_neurons_.d_spike_rate_target);
    release_mem(d_neurons_.d_a);
    release_mem(d_neurons_.d_b);
    release_mem(d_neurons_.d_tau_w);
    release_mem(d_neurons_.d_v_rest);
    release_mem(d_neurons_.d_neuron_kind);
    release_mem(d_neurons_.d_neuron_kind_target);

    release_mem(d_csr_csc_.d_csr_row_ptr);
    release_mem(d_csr_csc_.d_csr_col_idx);
    release_mem(d_csr_csc_.d_csc_row_ptr);
    release_mem(d_csr_csc_.d_csc_col_idx);
    release_mem(d_csr_csc_.d_csc_to_csr_index_mapping);
    release_mem(d_csr_csc_.d_weight);
    release_mem(d_csr_csc_.d_delay);
    release_mem(d_csr_csc_.d_potentiation_bias);
    release_mem(d_csr_csc_.d_depression_bias);

    release_mem(d_pipeline_.d_delayed_charge);
    release_mem(d_pipeline_.d_dirty_delayed_charge);
    release_mem(d_pipeline_.d_spiked);
    release_mem(d_pipeline_.d_spiked_only);
    release_mem(d_pipeline_.d_spiked_only_offset);
    release_mem(d_pipeline_.d_trace_lut);
    release_mem(d_pipeline_.d_input);
    release_mem(d_pipeline_.d_output);

    d_global_ = {};
    d_neurons_ = {};
    d_csr_csc_ = {};
    d_pipeline_ = {};
    h_organism_buffers_ = {};
}

void SNNEngine::releaseOpenCLKernelsAndPrograms()
{
    auto release_kernel = [](cl_kernel& kernel) {
        if (kernel) {
            clReleaseKernel(kernel);
            kernel = nullptr;
        }
    };
    auto release_program = [](cl_program& program) {
        if (program) {
            clReleaseProgram(program);
            program = nullptr;
        }
    };

    release_kernel(delievery_leak_fire_kernel_);
    release_program(delievery_leak_fire_program_);
    release_kernel(input_to_spiked_kernel_);
    release_program(input_to_spiked_program_);
    release_kernel(compaction_kernel_);
    release_program(compaction_program_);
    release_kernel(stdp_kernel_);
    release_program(stdp_program_);
    release_kernel(propagation_kernel_);
    release_program(propagation_program_);
    release_kernel(homeostasis_kernel_);
    release_program(homeostasis_program_);
}

