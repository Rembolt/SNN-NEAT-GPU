#include "HostOrganismBuffers.h"
#include "App.h"
#include "Constants.h"
#include "util/Log.h"

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

namespace {

using SteadyClock = std::chrono::steady_clock;
using SteadyDuration = SteadyClock::duration;

void writeTickTimingDebugLog(
    unsigned long tick_index,
    SteadyDuration tick_elapsed,
    SteadyDuration tick_target_duration)
{
    if constexpr (!debug::k_debug_logs) {
        return;
    }

    const std::chrono::duration<double, std::milli> tick_run_ms = tick_elapsed;
    const std::chrono::duration<double, std::milli> tick_target_ms = tick_target_duration;
    const double tick_run_fps = tick_run_ms.count() > 0.0
        ? 1000.0 / tick_run_ms.count()
        : 0.0;
    std::ostringstream timing_status;
    timing_status << "[tick " << tick_index << " timing]"
                  << " run_ms=" << tick_run_ms.count()
                  << " target_ms=" << tick_target_ms.count()
                  << " run_fps=" << tick_run_fps << '\n';
    Util::Logs::write(timing_status.str());
}

void runSchedulerIteration(
    unsigned long tick_index,
    SteadyDuration tick_target_duration,
    SteadyClock::time_point tick_start_time,
    SteadyClock::time_point& next_tick_start,
    std::chrono::duration<double>& tick_average_duration)
{
    const auto tick_elapsed_time = SteadyClock::now() - tick_start_time;
    tick_average_duration = 0.05 * tick_elapsed_time + 0.95 * tick_average_duration;

    writeTickTimingDebugLog(tick_index, tick_elapsed_time, tick_target_duration);

    next_tick_start += tick_target_duration;
    const auto now = SteadyClock::now();
    if (now < next_tick_start) {
        std::this_thread::sleep_until(next_tick_start);
    } else {
        next_tick_start = now;
    }
}

}  // namespace

App::App(SNNEngine& snn_engine)
    : snn_engine_(snn_engine)
{
}

SNNEngine& App::snn_engine()
{
    return snn_engine_;
}

void App::initializeInputCapture()
{
    if (input_capture_initialized_) {
        return;
    }

    mouse_output_.initialize(
        h_app_params_.x_Bins,
        h_app_params_.y_Bins,
        h_app_params_.h_mouse_output_axis_decay);
    mouse_.initialize();
    mouse_.initializeMonitors(monitors_);
    if (monitors_.empty()) {
        Util::Errors::stop("No monitors found");
    }
    current_monitor_idx_ =
        mouse_.getCurrentMonitor(monitors_, current_monitor_idx_);

    if (mode_ == AppMode::Live) {
        mouse_output_.initializeGhostOverlay(monitors_);
    }

    input_capture_initialized_ = true;
}

void App::initializeSnnKernels()
{
    if (!buffers_allocated_) {
        Util::Errors::stop("initializeSnnKernels: allocateBuffers must be called first");
    }

    snn_engine_.initializeDelieveryLeakFireKernel();
    snn_engine_.initializeInputToSpikedKernel();
    snn_engine_.initializeSTDPKernel();
    snn_engine_.initializeCompactionKernel();
    snn_engine_.initializePropagationKernel();
    snn_engine_.initializeHomeostasisKernel();
}

void App::initialize()
{
    initializeInputCapture();
    if (buffers_allocated_) {
        initializeSnnKernels();
    }
}

HostMouseInput App::gatherMouseInputTick()
{
    glfwPollEvents();

    mouse_.updateRawPosition();
    mouse_.updateClickState();

    if (mouse_.hasLeftMonitor(current_monitor_idx_, monitors_)) {
        const int new_monitor_idx =
            mouse_.getCurrentMonitor(monitors_, current_monitor_idx_);
        if (new_monitor_idx != current_monitor_idx_) {
            current_monitor_idx_ = new_monitor_idx;
        }
    }

    if (!monitors_.empty()) {
        const MonitorDisplayInfo& monitor_info =
            monitors_[static_cast<std::size_t>(current_monitor_idx_)];
        mouse_.getMonitorLocalNormalized(monitor_info);
    }

    return mouse_.h_mouse_input;
}

std::vector<HostMouseInput> App::recordedInputForTraining(
    unsigned long tick_count,
    SteadyDuration tick_target_duration,
    int clicks_required)
{
    std::vector<HostMouseInput> input_history;
    int click_count = 0;

    if (tick_count == 0u) {
        return input_history;
    }

    input_history.resize(tick_count);

    auto next_tick_start = SteadyClock::now();
    std::chrono::duration<double> tick_average_duration = tick_target_duration;

    for (unsigned long i = 0; i < tick_count; ++i) {
        const auto tick_start_time = SteadyClock::now();

        input_history[i] = gatherMouseInputTick();

        if(input_history[i].left_click_edge >= 0.5f || input_history[i].right_click_edge >= 0.5f) {
            click_count++;
        }

        runSchedulerIteration(
            i,
            tick_target_duration,
            tick_start_time,
            next_tick_start,
            tick_average_duration);
    }

    if(click_count < clicks_required) {
        input_history.clear();
        input_history.resize(0);
    }

    return input_history;
}

void App::tickForTraining(
    const std::vector<HostMouseInput>& input_history,
    unsigned long tick_count,
    SteadyDuration tick_target_duration,
    FitnessCallback fitness_callback,
    void* user_data)
{
    if (!buffers_allocated_) {
        Util::Errors::stop("tickForTraining: buffers not allocated");
    }
    if (input_history.size() != tick_count) {
        Util::Errors::stop(
            "tickForTraining: input_history.size() ("
            + std::to_string(input_history.size())
            + ") != tick_count ("
            + std::to_string(tick_count)
            + ')');
    }

    const double tick_target_ms =
        std::chrono::duration<double, std::milli>(tick_target_duration).count();

    for (unsigned long i = 0; i < tick_count; ++i) {
        const auto tick_start_time = SteadyClock::now();

        tick(static_cast<cl_long>(i), input_history[i]);

        const auto tick_elapsed_time = SteadyClock::now() - tick_start_time;
        const double tick_elapsed_ms =
            std::chrono::duration<double, std::milli>(tick_elapsed_time).count();

        writeTickTimingDebugLog(i, tick_elapsed_time, tick_target_duration);

        if (fitness_callback) {
            fitness_callback(
                user_data,
                i,
                last_spiked_only_count_,
                tick_elapsed_ms,
                tick_target_ms,
                error_feedback_.clickEdgeThisTick(),
                error_feedback_.getHostFeedbackInput(),
                mouse_output_.hostOutput());
        }
    }
}

void App::resetOrganismState()
{
    error_feedback_.reset();
    error_feedback_.setTemporalDecayTicks(h_app_params_.h_feedback_temporal_decay_ticks);
    mouse_output_.initialize(
        h_app_params_.x_Bins,
        h_app_params_.y_Bins,
        h_app_params_.h_mouse_output_axis_decay);
    last_spiked_only_count_ = 0;
    snn_engine_.setCurrentTick(0);
    if (!snn_engine_.clearPropagationSpikeRates()) {
        Util::Errors::stop("resetOrganismState: clearPropagationSpikeRates failed");
    }
}

void App::run(
    unsigned long tick_count,
    SteadyDuration tick_target_duration,
    FitnessCallback fitness_callback,
    void* user_data)
{
    auto next_tick_start = SteadyClock::now();
    std::chrono::duration<double> tick_average_duration = tick_target_duration;

    const double tick_target_ms =
        std::chrono::duration<double, std::milli>(tick_target_duration).count();

    for (unsigned long i = 0; i < tick_count; ++i) {
        const HostMouseInput mouse_input = gatherMouseInputTick();

        const auto tick_start_time = SteadyClock::now();
        tick(static_cast<cl_long>(i), mouse_input);
        const auto tick_elapsed_time = SteadyClock::now() - tick_start_time;
        const double tick_elapsed_ms =
            std::chrono::duration<double, std::milli>(tick_elapsed_time).count();

        runSchedulerIteration(
            i,
            tick_target_duration,
            tick_start_time,
            next_tick_start,
            tick_average_duration);

        if (fitness_callback) {
            fitness_callback(
                user_data,
                i,
                last_spiked_only_count_,
                tick_elapsed_ms,
                tick_target_ms,
                error_feedback_.clickEdgeThisTick(),
                error_feedback_.getHostFeedbackInput(),
                mouse_output_.hostOutput());
        }
    }
}

void App::tick(cl_long tick_index, const HostMouseInput& mouse_input)
{
    snn_engine_.setCurrentTick(tick_index);

    if (mode_ != AppMode::TrainReplay && monitors_.empty()) {
        Util::Errors::stop("tick: no monitors");
    }

    error_feedback_.registerMouseInput(mouse_input);

    if (!monitors_.empty()) {
        uploadMouseInput(mouse_input);
    }

    if (!snn_engine_.inputToSpiked()) {
        Util::Errors::stop("tick: inputToSpiked failed");
    }
    if (!snn_engine_.clearOutputBuffer()) {
        Util::Errors::stop("tick: clearOutputBuffer before propagation failed");
    }
    if (!snn_engine_.uploadDelieveryLeakFireKernel()) {
        //apply adex to neruons recieving charge and mark as spiked if threshold is reached
        Util::Errors::stop("tick: uploadDelieveryLeakFireKernel failed");
    }
    if (!snn_engine_.uploadCompactionKernel()) {
        //warp atomics compaction spiked list into spiked only list
        Util::Errors::stop("tick: uploadCompactionKernel failed");
    }
    if (!snn_engine_.readSpikedOnlyTotalCount(last_spiked_only_count_)) {
        last_spiked_only_count_ = 0;
    }
    if (!snn_engine_.uploadSTDPKernel(last_spiked_only_count_)) {
        //teach neurons what to learn (affected by error feedback)
        Util::Errors::stop("tick: uploadSTDPKernel failed");
    }
    if (!snn_engine_.uploadPropagationKernel(last_spiked_only_count_)) {
        //propagate spikes + apply input weights + increment spiked output targets
        Util::Errors::stop("tick: uploadPropagationKernel failed");
    }
    // Kernel gates work to multiples of `ticks_between_homeostasis`; enqueue every tick so
    if (!snn_engine_.uploadHomeostasisKernel()) {
        Util::Errors::stop("tick: uploadHomeostasisKernel failed");
    }

    snn_engine_.getOutputBuffer(h_organism_buffers_.h_pipeline_buffers_.h_output);
    //calculate normalized output
    mouse_output_.updateOutput(
        h_organism_buffers_.h_pipeline_buffers_.h_output,
        h_organism_buffers_.h_global_params_.h_output_slot_neuron_counts);

    //send output mouse to error feedback
    error_feedback_.registerSNNOutput(mouse_output_.hostOutput());
    error_feedback_.updateErrorFeedback(static_cast<unsigned long>(tick_index));
    uploadErrorFeedback(error_feedback_.getHostFeedbackInput());

    if (mode_ == AppMode::Live && !monitors_.empty()) {
        const auto& active_monitor =
            monitors_[static_cast<std::size_t>(current_monitor_idx_)];
        const HostMouseOutput& snn_out = mouse_output_.hostOutput();
        const double predicted_global_x =
            static_cast<double>(active_monitor.virtual_pos_x)
            + static_cast<double>(snn_out.predicted_x)
                * static_cast<double>(active_monitor.virtual_width);
        const double predicted_global_y =
            static_cast<double>(active_monitor.virtual_pos_y)
            + static_cast<double>(snn_out.predicted_y)
                * static_cast<double>(active_monitor.virtual_height);
        mouse_output_.renderGhostOverlayFrame(
            active_monitor,
            predicted_global_x,
            predicted_global_y);
    }

    if (debug::k_debug_logs) {
        const HostMouseOutput& snn_out = mouse_output_.hostOutput();
        std::ostringstream output_status;
        output_status << "[tick " << tick_index << " output]"
                      << " predicted=(" << snn_out.predicted_x << ',' << snn_out.predicted_y
                      << ") certainty=" << snn_out.certainty
                      << " clicks(r,l)=(" << snn_out.right_click << ',' << snn_out.left_click
                      << ")\n";
        Util::Logs::write(output_status.str());
    }
}

void App::allocateBuffers(
    cl_device_id       device,
    cl_context         context,
    cl_command_queue   queue,
    const HostOrganismBuffers& organism_buffers,
    const HostAppParams& app_params)
{
    if (buffers_allocated_) {
        Util::Errors::stop(
            "allocateBuffers: buffers already allocated; call freeBuffers first");
    }

    device_ = device;
    queue_ = queue;
    context_ = context;
    h_organism_buffers_ = organism_buffers;
    h_app_params_ = app_params;

    error_feedback_.setTemporalDecayTicks(h_app_params_.h_feedback_temporal_decay_ticks);
    mouse_output_.initialize(
        h_app_params_.x_Bins,
        h_app_params_.y_Bins,
        h_app_params_.h_mouse_output_axis_decay);

    const std::size_t expected_input_floats =
        static_cast<std::size_t>(h_app_params_.h_screen_input_size)
        + static_cast<std::size_t>(h_app_params_.h_mouse_input_size)
        + static_cast<std::size_t>(h_app_params_.h_error_input_size);
    if (h_organism_buffers_.h_pipeline_buffers_.h_input.size() != expected_input_floats) {
        Util::Errors::stop(
            "allocateBuffers: pipeline h_input size mismatch (got "
            + std::to_string(h_organism_buffers_.h_pipeline_buffers_.h_input.size())
            + ", expected screen+mouse+error = "
            + std::to_string(expected_input_floats) + ")");
    }
    if (h_organism_buffers_.h_global_params_.h_num_of_input_neurons
        != static_cast<uint32_t>(expected_input_floats)) {
        Util::Errors::stop(
            "allocateBuffers: h_num_of_input_neurons mismatch (got "
            + std::to_string(h_organism_buffers_.h_global_params_.h_num_of_input_neurons)
            + ", expected "
            + std::to_string(expected_input_floats) + ")");
    }

    snn_engine_.allocateBuffers(device, context, queue, h_organism_buffers_);
    buffers_allocated_ = true;
}

void App::uploadMouseInput(const HostMouseInput& h_mouse_input)
{
    std::vector<float> h_mouse_input_buffer =
        Util::packHostInputStructToFloatVector(h_mouse_input);
    if (h_mouse_input_buffer.size() != static_cast<std::size_t>(h_app_params_.h_mouse_input_size)) {
        Util::Errors::stop(
            "uploadMouseInput: HostMouseInput packed size does not match h_mouse_input_size from DNAReader");
    }
    const std::size_t h_mouse_input_offset =
        static_cast<std::size_t>(h_app_params_.h_screen_input_size);
    snn_engine_.uploadToInputBuffer(
        h_mouse_input_offset,
        h_mouse_input_buffer);
}

void App::uploadErrorFeedback(const HostFeedbackInput& h_feedback_input)
{
    std::vector<float> h_error_feedback_input_buffer =
        Util::packHostInputStructToFloatVector(h_feedback_input);
    if (h_error_feedback_input_buffer.size() != static_cast<std::size_t>(h_app_params_.h_error_input_size)) {
        Util::Errors::stop(
            "uploadErrorFeedback: HostFeedbackInput packed size does not match h_error_input_size from DNAReader");
    }
    const std::size_t h_error_feedback_offset =
        static_cast<std::size_t>(h_app_params_.h_screen_input_size + h_app_params_.h_mouse_input_size);
    snn_engine_.uploadToInputBuffer(h_error_feedback_offset, h_error_feedback_input_buffer);
}

App::~App()
{
    freeBuffers();
}

void App::freeBuffers()
{
    if (!buffers_allocated_) {
        return;
    }

    snn_engine_.freeDeviceBuffers();
    h_app_params_ = {};
    h_organism_buffers_ = {};
    buffers_allocated_ = false;
}
