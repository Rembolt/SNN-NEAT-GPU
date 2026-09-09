#pragma once

/*

Neuron param 1d buffers:
-Potential current charge (V)
-Adaptation current (w)
-Subthreshold adaptation strength (a)
-Post Spike-Triggered (w) increment (b)
-Adaptation time constant (t_w)
-threshold (V_Threshold)
-Resting potential (V_rest)
-Last AdEx updated tick
-Refractory end tick
-Last fire tick(used to calculate trace)
-Fire quantity since last homeostasis
-homeostasis target spike rate
-neuron type(index for output neuron preferences)
-Refractory length

organization buffers:
-current tick
-CSR column index 
-CSR row pointer 
-CSC column index
-CSC row pointer 
-CSC_to_CSR index mapping
-weights list
-delays list
-potentiation bias list
-depression bias list
-circular updated list dirty bit(2d, N/32 | max delay)(who recieved charge)
-spiked list(becomes prefix sum, then spike only list)
-spiked only list
-spiked only list offset
-circular delay list(2d, column size = max delay)
-input buffer()
-output buffer ()

single global values (CONSTANT SHARED MEMORY)
-Num of Neurons
-desired rate
-max trace tick
-max delay
-trace decay look-up table
-Homeostasis weight mod.
-Homeostasis pot. bias mod.
-Homeostasis dep. bias mod.
-Homeostasis threshold mod.
-Homeostasis rate max distance
-Ticks between homeostasis
-Membrane Capacitance (C)
-Leak conductance (g_Leak)
-Leak reversal potential (resting) (E_Leak)
-Slope factor (spike sharpness) (delta_T)
-Spike peak voltage (V_peak)

AM I missing any?
*/

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>

#include "HostOrganismBuffers.h"
#include "util/Log.h"

struct DeviceGlobalParams {
    cl_long  d_current_tick;
    cl_uint  d_words_per_row;
    cl_uint  d_num_of_input_neurons;
    cl_uint  d_num_output_slots;
    cl_uint  d_feedback_temporal_decay_index;
    cl_uint  d_last_click_distance_index;
    cl_float d_trace_tau;
    cl_uint  d_num_neurons;
    cl_uint  d_num_synapses;
    cl_uint  d_max_trace_len;
    cl_uint  d_max_delay;
    cl_float d_click_distance_weight;
    cl_float d_homeo_weight_mod;
    cl_float d_homeo_pot_mod;
    cl_float d_homeo_dep_mod;
    cl_float d_homeo_thresh_mod;
    cl_float d_homeo_rate_max_distance;
    cl_uint  d_ticks_between_homeostasis;
    cl_float d_fire_rate_tau;
    cl_float d_capacitance_C;
    cl_float d_leak_g;
    cl_float d_leak_E;
    cl_float d_slope_delta_T;
    cl_float d_v_peak;
    cl_float d_dt_ms;
    cl_mem   d_inhibitory_spike_rate;
    cl_mem   d_excitatory_spike_rate;
    cl_mem   d_output_slot_neuron_counts;
};

struct DeviceNeuronBuffers {
    cl_mem d_V;
    cl_mem d_W;
    cl_mem d_threshold_V;
    cl_mem d_refract_end;
    cl_mem d_refractory_length;
    cl_mem d_last_delievery_update;
    cl_mem d_last_spike;
    cl_mem d_spike_rate;
    cl_mem d_spike_rate_target;
    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_tau_w;
    cl_mem d_v_rest;
    cl_mem d_neuron_kind;
    cl_mem d_neuron_kind_target;

};

struct DeviceCSRCSCBuffers {
    cl_mem d_csr_row_ptr;
    cl_mem d_csr_col_idx;
    cl_mem d_csc_row_ptr;
    cl_mem d_csc_col_idx;
    cl_mem d_csc_to_csr_index_mapping;
    cl_mem d_weight;
    cl_mem d_delay;
    cl_mem d_potentiation_bias;
    cl_mem d_depression_bias;
};

struct DevicePipelineBuffers {
    cl_mem d_delayed_charge;
    cl_mem d_dirty_delayed_charge;
    cl_mem d_spiked;
    cl_mem d_spiked_only;
    cl_mem d_spiked_only_offset;
    cl_mem d_trace_lut;
    cl_mem d_input;
    cl_mem d_output;
};



class SNNEngine {
public:
    SNNEngine();
    ~SNNEngine();

    
    void allocateBuffers(
        cl_device_id        device,
        cl_context          context,
        cl_command_queue    queue,
        const HostOrganismBuffers& organism_buffers);

    /// Compile once; no-op if kernel already exists. Args are bound in upload*.
    void initializeInputToSpikedKernel();
    void initializeDelieveryLeakFireKernel();
    void initializeCompactionKernel();
    void initializeSTDPKernel();
    void initializePropagationKernel();
    void initializeHomeostasisKernel();

    bool clearOutputBuffer();
    bool inputToSpiked();
    bool uploadDelieveryLeakFireKernel();
    bool uploadCompactionKernel();
    bool uploadSTDPKernel(std::uint32_t spiked_only_count);
    bool uploadPropagationKernel(std::uint32_t spiked_only_count);
    bool uploadHomeostasisKernel();
    /// Releases OpenCL memory only; keep compiled kernels across organisms.
    void freeDeviceBuffers();
    /// Releases kernel handles and compiled programs (not device memory).
    void releaseOpenCLKernelsAndPrograms();
    void uploadToInputBuffer(size_t offset, const std::vector<float>& h_input_buffer);

    cl_command_queue getCommandQueue() const {
        return queue_;
    }
    const HostOrganismBuffers& getHostOrganismBuffers() const {
        return h_organism_buffers_;
    }
    const DeviceGlobalParams& getDeviceGlobalParams() const {
        return d_global_;
    }
    const DeviceNeuronBuffers& getDeviceNeuronBuffers() const {
        return d_neurons_;
    }
    const DeviceCSRCSCBuffers& getDeviceCSRCSCBuffers() const {
        return d_csr_csc_;
    }
    const DevicePipelineBuffers& getDevicePipelineBuffers() const {
        return d_pipeline_;
    }
    
    //setters and getters
    void setCurrentTick(cl_long tick_index){
        d_global_.d_current_tick = tick_index;
    }

    cl_long getCurrentTick() const {
        return d_global_.d_current_tick;
    }
    cl_mem getInputBuffer() const {
        if (!queue_ || !d_pipeline_.d_input) {
            Util::Errors::stop("getInputBuffer: missing OpenCL queue or d_input buffer\n");
        }
        return d_pipeline_.d_input;
    }
    void getOutputBuffer(std::vector<uint32_t>& h_output);

    /// Waits for queued work, then reads compaction spike list length for the most recent compaction pass.
    bool readSpikedOnlyTotalCount(std::uint32_t& out_count);
    bool readPropagationSpikeRates(
        std::uint32_t& out_inhibitory,
        std::uint32_t& out_excitatory);
    bool clearPropagationSpikeRates();

private:
    cl_context       context_{};
    cl_command_queue queue_{};
    cl_device_id     device_{};
    cl_program       delievery_leak_fire_program_{};
    cl_kernel        delievery_leak_fire_kernel_{};
    cl_program       input_to_spiked_program_{};
    cl_kernel        input_to_spiked_kernel_{};
    cl_program       compaction_program_{};
    cl_kernel        compaction_kernel_{};
    cl_program       stdp_program_{};
    cl_kernel        stdp_kernel_{};
    cl_program       propagation_program_{};
    cl_kernel        propagation_kernel_{};
    cl_program       homeostasis_program_{};
    cl_kernel        homeostasis_kernel_{};
    HostOrganismBuffers  h_organism_buffers_{};
    DeviceGlobalParams       d_global_{};
    DeviceNeuronBuffers      d_neurons_{};
    DeviceCSRCSCBuffers      d_csr_csc_{};
    DevicePipelineBuffers    d_pipeline_{};

};
