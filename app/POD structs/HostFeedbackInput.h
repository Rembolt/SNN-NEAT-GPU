#pragma once

/*
Packed float slice uploaded after kernels; consumed on the following tick (see App::uploadErrorFeedback).
Uses HostMouseInput x_norm/y_norm vs HostMouseOutput predicted_x/predicted_y.
*/

struct HostFeedbackInput {
    // STDP / click snapshot (click-slot fields refresh on physical click edge only; slice still uploaded every tick)
    float last_click_prediction_error;// [0,1] Euclidean vs envelopes, clamped (max raw sqrt(2) before clamp)
    float last_click_prediction_temporal_decay;// [0,1] decay factor: ~1 right after click, ~0 after temporal_decay_ticks

    float last_click_prediction_error_X_pos;// [0,1] envelope per axis toward +x from click-vs-predicted delta
    float last_click_prediction_error_X_neg;// [0,1]
    float last_click_prediction_error_Y_pos;// [0,1]
    float last_click_prediction_error_Y_neg;// [0,1]
    float last_click_prediction_error_delta;// signed step vs previous last_click_prediction_error (not clamped)
    float last_click_predicted_left_click;// [0,1] event gate + stronger left_click logit at click
    float last_click_predicted_right_click;// [0,1]
    float last_click_was_left_click;// [0,1]
    float last_click_was_right_click;// [0,1]

    float last_click_distance_X_pos;// [0,1] last click vs current predicted position (envelopes)
    float last_click_distance_X_neg;// [0,1]
    float last_click_distance_Y_pos;// [0,1]
    float last_click_distance_Y_neg;// [0,1]
    float last_click_distance;// [0,1] Euclidean vs envelopes, clamped
    float last_click_distance_delta;// signed step vs previous last_click_distance

    float predicted_to_real_distance_X_pos;// [0,1] predicted vs real cursor (envelopes)
    float predicted_to_real_distance_X_neg;// [0,1]
    float predicted_to_real_distance_Y_pos;// [0,1]
    float predicted_to_real_distance_Y_neg;// [0,1]
    float predicted_to_real_distance;// [0,1] Euclidean vs envelopes, clamped
    float predicted_to_real_distance_delta;// signed step vs previous predicted_to_real_distance

    // Per-tick motion: velocity from envelope deltas, acceleration/jerk from vector deltas (split into pos/neg per axis)
    float delta_x_pos;// [0,1]
    float delta_x_neg;// [0,1]
    float delta_y_pos;// [0,1]
    float delta_y_neg;// [0,1]
    float speed_magnitude;// [0,1] clamped length of velocity (vx,vy) from envelopes
    float acceleration_x_pos;// [0,1]
    float acceleration_x_neg;// [0,1]
    float acceleration_y_pos;// [0,1]
    float acceleration_y_neg;// [0,1]
    float acceleration_magnitude;// [0,1] clamped |a|
    float jerk_x_pos;// [0,1]
    float jerk_x_neg;// [0,1]
    float jerk_y_pos;// [0,1]
    float jerk_y_neg;// [0,1]
    float jerk_magnitude;// [0,1] clamped |j|

    // vx*acc_y - vy*acc_x with acc = v_t - v_{t-1}; straight motion -> ~0 on both channels
    // Positive scalar -> curvature_right; negative -> curvature_left (screen-axis handedness)
    float curvature_left;// [0,1]
    float curvature_right;// [0,1]

    // Direction trig from velocity angle; idle -> all zero (no remapped neutral 0.5)
    float delta_cosine_pos;// [0,1]
    float delta_cosine_neg;// [0,1]
    float delta_sine_pos;// [0,1]
    float delta_sine_neg;// [0,1]

    float left_mouse_hold_duration;
    float right_mouse_hold_duration;
    float mouse_idle_duration;
};
