
#pragma once

#include <cstdint>

struct HostAppParams {
    /// Reserved prefix of `h_input`; always 0 without screen vision features.
    int32_t h_screen_input_size{};
    int32_t h_mouse_input_size{};
    int32_t h_error_input_size{};
    std::uint32_t h_feedback_temporal_decay_ticks{};
    float h_mouse_output_axis_decay{};
    int32_t x_Bins{};
    int32_t y_Bins{};
};
