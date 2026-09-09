#pragma once

#include <cstdint>
#include <vector>

struct HostMouseOutput {
    // axis bins to allow SNN to predict multiple places mouse could click
    std::vector<float> x;
    int x_length{};
    std::vector<float> y;
    int y_length{};

    float certainty{};//[0,1]
    float output_axis_decay{};//[0,1]

    // highest voted bin gets interpolated with adjacent bins for sub-bin precision
    float predicted_x{};//[0,1] NORMALIZED
    float predicted_y{};//[0,1] NORMALIZED

    float left_click{};//[0,1] NORMALIZED, optional trailing output slots
    float right_click{};//[0,1] NORMALIZED
};
