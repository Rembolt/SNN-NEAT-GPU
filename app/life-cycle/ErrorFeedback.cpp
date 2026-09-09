#include "ErrorFeedback.h"

#include <algorithm>
#include <cmath>
#include <limits>

float ErrorFeedback::clamp01(float v) noexcept
{
    return std::clamp(v, 0.0f, 1.0f);
}

ErrorFeedback::ErrorFeedback()
    : last_physical_click_tick_(std::numeric_limits<unsigned long>::max())
{
}

void ErrorFeedback::setTemporalDecayTicks(std::uint32_t ticks) noexcept
{
    temporal_decay_ticks_ = std::max<std::uint32_t>(1u, ticks);
}

void ErrorFeedback::reset()
{
    *this = ErrorFeedback{};
}

void ErrorFeedback::registerMouseInput(const HostMouseInput& h_mouse_input)
{
    h_mouse_input_ = h_mouse_input;
}

void ErrorFeedback::registerSNNOutput(const HostMouseOutput& h_mouse_output)
{
    h_mouse_output_ = h_mouse_output;
}

void ErrorFeedback::updateErrorFeedback(unsigned long current_tick)
{
    current_tick_ = current_tick;
    click_edge_this_tick_ = false;

    const bool left_now = h_mouse_input_.left_mouse_clicked >= 0.5f;
    const bool right_now = h_mouse_input_.right_mouse_clicked >= 0.5f;

    if (h_mouse_input_.left_click_edge >= 0.5f || h_mouse_input_.right_click_edge >= 0.5f) {
        last_physical_click_tick_ = current_tick_;
        had_physical_click_ = true;

        const float dx_click =
            h_mouse_input_.x_norm - h_mouse_output_.predicted_x;
        const float dy_click =
            h_mouse_input_.y_norm - h_mouse_output_.predicted_y;

        h_feedback_input_.last_click_prediction_error_X_pos = clamp01(dx_click);
        h_feedback_input_.last_click_prediction_error_X_neg = clamp01(-dx_click);
        h_feedback_input_.last_click_prediction_error_Y_pos = clamp01(dy_click);
        h_feedback_input_.last_click_prediction_error_Y_neg = clamp01(-dy_click);

        const float new_click_err = clamp01(std::sqrt(
            std::pow(h_feedback_input_.last_click_prediction_error_X_pos
                         - h_feedback_input_.last_click_prediction_error_X_neg,
                     2.F)
            + std::pow(h_feedback_input_.last_click_prediction_error_Y_pos
                           - h_feedback_input_.last_click_prediction_error_Y_neg,
                       2.F)));

        h_feedback_input_.last_click_prediction_error_delta =
            new_click_err - h_feedback_input_.last_click_prediction_error;
        h_feedback_input_.last_click_prediction_error = new_click_err;

        h_feedback_input_.last_click_predicted_left_click =
            left_now && (h_mouse_output_.left_click > h_mouse_output_.right_click) ? 1.0f : 0.0f;
        h_feedback_input_.last_click_predicted_right_click =
            right_now && (h_mouse_output_.right_click > h_mouse_output_.left_click) ? 1.0f : 0.0f;
        h_feedback_input_.last_click_was_left_click = left_now ? 1.0f : 0.0f;
        h_feedback_input_.last_click_was_right_click = right_now ? 1.0f : 0.0f;

        last_click_h_mouse_input_ = h_mouse_input_;
        click_edge_this_tick_ = true;
    }

    if (had_physical_click_
        && last_physical_click_tick_ != std::numeric_limits<unsigned long>::max()) {
        const unsigned long age = current_tick_ - last_physical_click_tick_;
        h_feedback_input_.last_click_prediction_temporal_decay =
            clamp01(1.0f - static_cast<float>(age)
                           / static_cast<float>(temporal_decay_ticks_));
    } else {
        h_feedback_input_.last_click_prediction_temporal_decay = 0.0f;
    }

    if (had_physical_click_) {
        const float lx = last_click_h_mouse_input_.x_norm;
        const float ly = last_click_h_mouse_input_.y_norm;
        h_feedback_input_.last_click_distance_X_pos =
            clamp01(lx - h_mouse_output_.predicted_x);
        h_feedback_input_.last_click_distance_X_neg =
            clamp01(-(lx - h_mouse_output_.predicted_x));
        h_feedback_input_.last_click_distance_Y_pos =
            clamp01(ly - h_mouse_output_.predicted_y);
        h_feedback_input_.last_click_distance_Y_neg =
            clamp01(-(ly - h_mouse_output_.predicted_y));

        const float new_last_click_distance = clamp01(std::sqrt(
            std::pow(h_feedback_input_.last_click_distance_X_pos
                         - h_feedback_input_.last_click_distance_X_neg,
                     2.F)
            + std::pow(h_feedback_input_.last_click_distance_Y_pos
                           - h_feedback_input_.last_click_distance_Y_neg,
                       2.F)));

        h_feedback_input_.last_click_distance_delta =
            new_last_click_distance - h_feedback_input_.last_click_distance;
        h_feedback_input_.last_click_distance = new_last_click_distance;
    } else {
        h_feedback_input_.last_click_distance_delta = 0.0f;
        h_feedback_input_.last_click_distance = 0.0f;
    }

    const float pr_dx = h_mouse_output_.predicted_x - h_mouse_input_.x_norm;
    const float pr_dy = h_mouse_output_.predicted_y - h_mouse_input_.y_norm;
    h_feedback_input_.predicted_to_real_distance_X_pos = clamp01(pr_dx);
    h_feedback_input_.predicted_to_real_distance_X_neg = clamp01(-pr_dx);
    h_feedback_input_.predicted_to_real_distance_Y_pos = clamp01(pr_dy);
    h_feedback_input_.predicted_to_real_distance_Y_neg = clamp01(-pr_dy);

    const float new_predicted_to_real_distance = clamp01(std::sqrt(
        std::pow(h_feedback_input_.predicted_to_real_distance_X_pos
                     - h_feedback_input_.predicted_to_real_distance_X_neg,
                 2.F)
        + std::pow(h_feedback_input_.predicted_to_real_distance_Y_pos
                       - h_feedback_input_.predicted_to_real_distance_Y_neg,
                   2.F)));

    h_feedback_input_.predicted_to_real_distance_delta =
        new_predicted_to_real_distance - h_feedback_input_.predicted_to_real_distance;
    h_feedback_input_.predicted_to_real_distance = new_predicted_to_real_distance;

    const float new_delta_x_pos =
        clamp01(h_mouse_input_.x_norm - last_h_mouse_input_.x_norm);
    const float new_delta_x_neg =
        clamp01(-(h_mouse_input_.x_norm - last_h_mouse_input_.x_norm));
    const float new_delta_y_pos =
        clamp01(h_mouse_input_.y_norm - last_h_mouse_input_.y_norm);
    const float new_delta_y_neg =
        clamp01(-(h_mouse_input_.y_norm - last_h_mouse_input_.y_norm));

    const float prev_vx =
        h_feedback_input_.delta_x_pos - h_feedback_input_.delta_x_neg;
    const float prev_vy =
        h_feedback_input_.delta_y_pos - h_feedback_input_.delta_y_neg;
    const float vx = new_delta_x_pos - new_delta_x_neg;
    const float vy = new_delta_y_pos - new_delta_y_neg;

    const float new_speed_magnitude =
        clamp01(std::sqrt(std::pow(new_delta_x_pos - new_delta_x_neg, 2.F)
                          + std::pow(new_delta_y_pos - new_delta_y_neg, 2.F)));

    const float acc_x = vx - prev_vx;
    const float acc_y = vy - prev_vy;

    const float prev_acc_x =
        h_feedback_input_.acceleration_x_pos - h_feedback_input_.acceleration_x_neg;
    const float prev_acc_y =
        h_feedback_input_.acceleration_y_pos - h_feedback_input_.acceleration_y_neg;

    const float jerk_x = acc_x - prev_acc_x;
    const float jerk_y = acc_y - prev_acc_y;

    const float new_acceleration_x_pos = clamp01(std::max(0.F, acc_x));
    const float new_acceleration_x_neg = clamp01(std::max(0.F, -acc_x));
    const float new_acceleration_y_pos = clamp01(std::max(0.F, acc_y));
    const float new_acceleration_y_neg = clamp01(std::max(0.F, -acc_y));
    const float new_acceleration_magnitude =
        clamp01(std::sqrt(acc_x * acc_x + acc_y * acc_y));

    const float new_jerk_x_pos = clamp01(std::max(0.F, jerk_x));
    const float new_jerk_x_neg = clamp01(std::max(0.F, -jerk_x));
    const float new_jerk_y_pos = clamp01(std::max(0.F, jerk_y));
    const float new_jerk_y_neg = clamp01(std::max(0.F, -jerk_y));
    const float new_jerk_magnitude =
        clamp01(std::sqrt(jerk_x * jerk_x + jerk_y * jerk_y));

    h_feedback_input_.delta_x_pos = new_delta_x_pos;
    h_feedback_input_.delta_x_neg = new_delta_x_neg;
    h_feedback_input_.delta_y_pos = new_delta_y_pos;
    h_feedback_input_.delta_y_neg = new_delta_y_neg;
    h_feedback_input_.speed_magnitude = new_speed_magnitude;
    h_feedback_input_.acceleration_x_pos = new_acceleration_x_pos;
    h_feedback_input_.acceleration_x_neg = new_acceleration_x_neg;
    h_feedback_input_.acceleration_y_pos = new_acceleration_y_pos;
    h_feedback_input_.acceleration_y_neg = new_acceleration_y_neg;
    h_feedback_input_.acceleration_magnitude = new_acceleration_magnitude;
    h_feedback_input_.jerk_x_pos = new_jerk_x_pos;
    h_feedback_input_.jerk_x_neg = new_jerk_x_neg;
    h_feedback_input_.jerk_y_pos = new_jerk_y_pos;
    h_feedback_input_.jerk_y_neg = new_jerk_y_neg;
    h_feedback_input_.jerk_magnitude = new_jerk_magnitude;

    const float curvature_raw = vx * acc_y - vy * acc_x;
    h_feedback_input_.curvature_right = clamp01(std::max(0.F, curvature_raw));
    h_feedback_input_.curvature_left = clamp01(std::max(0.F, -curvature_raw));

    if (new_speed_magnitude < 0.01f) {
        h_feedback_input_.delta_cosine_pos = 0.F;
        h_feedback_input_.delta_cosine_neg = 0.F;
        h_feedback_input_.delta_sine_pos = 0.F;
        h_feedback_input_.delta_sine_neg = 0.F;
    } else {
        const float angle = std::atan2(vy, vx);
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        h_feedback_input_.delta_cosine_pos = clamp01(std::max(0.F, c));
        h_feedback_input_.delta_cosine_neg = clamp01(std::max(0.F, -c));
        h_feedback_input_.delta_sine_pos = clamp01(std::max(0.F, s));
        h_feedback_input_.delta_sine_neg = clamp01(std::max(0.F, -s));
    }

    if (left_now) {
        h_feedback_input_.left_mouse_hold_duration += 1.0f;
    } else {
        h_feedback_input_.left_mouse_hold_duration = 0.0f;
    }
    if (right_now) {
        h_feedback_input_.right_mouse_hold_duration += 1.0f;
    } else {
        h_feedback_input_.right_mouse_hold_duration = 0.0f;
    }

    if (h_feedback_input_.speed_magnitude < 0.01f) {
        h_feedback_input_.mouse_idle_duration += 1.0f;
    } else {
        h_feedback_input_.mouse_idle_duration = 0.0f;
    }

    last_h_mouse_input_ = h_mouse_input_;
}
