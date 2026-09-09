#pragma once

/*
Compares HostMouseInput (normalized to monitor) with HostMouseOutput predicted_x/predicted_y each tick after decode
Feedback is packed into HostFeedbackInput and uploaded for the next tick's STDP/input path
Temporal decay uses ticks since last physical click vs temporal_decay_ticks (factor in [0,1], not raw age)
*/

#include "HostFeedbackInput.h"
#include "HostMouseInput.h"
#include "HostMouseOutput.h"

#include <cstdint>

class ErrorFeedback {
public:
    ErrorFeedback();

    void setTemporalDecayTicks(std::uint32_t ticks) noexcept;

    void reset();

    void registerMouseInput(const HostMouseInput& h_mouse_input);
    void registerSNNOutput(const HostMouseOutput& h_mouse_output);

    /// Uses registered mouse + SNN output for this tick; call once per tick after outputs are ready.
    void updateErrorFeedback(unsigned long current_tick);

    [[nodiscard]] const HostFeedbackInput& getHostFeedbackInput() const noexcept
    {
        return h_feedback_input_;
    }

    /// Valid for the current tick after `updateErrorFeedback` returns.
    [[nodiscard]] bool clickEdgeThisTick() const noexcept { return click_edge_this_tick_; }

private:
    static float clamp01(float v) noexcept;


    HostFeedbackInput h_feedback_input_{};
    HostMouseInput h_mouse_input_{};
    HostMouseOutput h_mouse_output_{};

    HostMouseInput last_h_mouse_input_{};
    HostMouseInput last_click_h_mouse_input_{};

    unsigned long current_tick_{};
    unsigned long last_physical_click_tick_{};
    bool had_physical_click_{false};
    bool click_edge_this_tick_{false};
    std::uint32_t temporal_decay_ticks_{120};
};
