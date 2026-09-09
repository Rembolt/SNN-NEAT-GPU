#pragma once
/*
tick N:
  [CPU]         clear pipeline buffers              -> last tick delay, dirty last tick delay, spiked_list, spike_only_list, spiked only list offset
  [CPU] 	    call mouse capture from last frame	-> glfw + normalize + input Buffer + errorfeedback history
  [CPU]    	    error feedback for last frame		-> call NEAT class + Buffer
  [CPU]  	    input pipeline 			            -> DXGI + [Compute Shader] downsample + normalize + diff + Buffer
  [kernel]  	delivery + decay + fire + reset     -> delay_charge + AdEx + writes spike_bits (mark spiked_list)
  [kernel]  	compaction                          -> spike_list → spiked_only_list with spike_count (warp-agg atomics)
  [kernel]  	propagation                         -> reads spike_only_list, writes to delay circular buffer[target_slot], and mark ist dirty bit delay version
  [kernel]  	STDP                                -> reads spike_only_list, writes reverse CSR weights
  [kernel]  	homeostasis (every 1000 ticks)      -> mod threshold + mod pot bias + mod dep bias
  [CPU] 	    send output info                    -> errorfeedback history
  [CPU]     	tick++


input upload layout:
  [0 .. screen_input_size-1]                       reserved (unused; size 0)
  [screen_input_size .. + mouse_input_size-1]      HostMouseInput fields
  [screen_input_size + mouse_input_size .. end]    HostFeedbackInput fields
*/
#include "MouseCapture.h"
#include "MonitorDisplayInfo.h"
#include "MouseOutput.h"
#include "SNNEngine.h"
#include "ErrorFeedback.h"
#include "HostAppParams.h"
#include <chrono>
#include <cstdint>
#include <vector>


enum class AppMode {
    Live,
    RecordInput,
    TrainReplay,
};

class App {
public:
    explicit App(SNNEngine& snn_engine);

    SNNEngine& snn_engine();

    void setMode(AppMode mode) noexcept { mode_ = mode; }
    [[nodiscard]] AppMode mode() const noexcept { return mode_; }

    [[nodiscard]] const HostMouseOutput& hostMouseOutput() const noexcept
    {
        return mouse_output_.hostOutput();
    }

    /// Mouse, monitors, and optional ghost overlay (Live only). Safe without GPU buffers.
    void initializeInputCapture();
    /// Compile OpenCL programs once (no-op after first); requires `allocateBuffers` first.
    void initializeSnnKernels();
    /// Input capture always; SNN kernels when buffers are already allocated.
    void initialize();

    [[nodiscard]] HostMouseInput gatherMouseInputTick();

    void tick(cl_long tick_index, const HostMouseInput& mouse_input);

    using FitnessCallback = void(*)(
        void* user_data,
        unsigned long tick_index,
        std::uint32_t spiked_only_count,
        double tick_elapsed_ms,
        double tick_target_ms,
        bool mouse_click_edge,
        const HostFeedbackInput& feedback,
        const HostMouseOutput& mouse_out);

    void run(
        unsigned long tick_count,
        std::chrono::steady_clock::duration tick_target_duration,
        FitnessCallback per_tick,
        void* user_data);

    [[nodiscard]] std::vector<HostMouseInput> recordedInputForTraining(
        unsigned long tick_count,
        std::chrono::steady_clock::duration tick_target_duration,
        int clicks_required);

    void tickForTraining(
        const std::vector<HostMouseInput>& input_history,
        unsigned long tick_count,
        std::chrono::steady_clock::duration tick_target_duration,
        FitnessCallback per_tick,
        void* user_data);

    /// Per-organism CPU/GPU run state after a new DNA buffer layout is allocated.
    void resetOrganismState();

    void allocateBuffers(
        cl_device_id        device,
        cl_context          context,
        cl_command_queue    queue,
        const HostOrganismBuffers& organism_buffers,
        const HostAppParams& app_params);
    void uploadMouseInput(const HostMouseInput& h_mouse_input);
    void uploadErrorFeedback(const HostFeedbackInput& h_feedback_input);
    void freeBuffers();
    ~App();

private:
    SNNEngine& snn_engine_;
    AppMode mode_{AppMode::Live};
    ErrorFeedback error_feedback_;
    std::uint32_t last_spiked_only_count_{};
    cl_context       context_{};
    cl_command_queue queue_{};
    cl_device_id     device_{};
    MouseCapture                    mouse_;
    MouseOutput                     mouse_output_;
    std::vector<MonitorDisplayInfo> monitors_;
    int                             current_monitor_idx_ = 0;
    bool                            input_capture_initialized_{false};

    HostOrganismBuffers  h_organism_buffers_{};
    HostAppParams         h_app_params_{};

    bool             buffers_allocated_{false};
};
