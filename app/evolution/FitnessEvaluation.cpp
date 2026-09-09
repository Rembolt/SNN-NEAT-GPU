#include "FitnessEvaluation.h"
#include "Constants.h"
#include "util/Log.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <sstream>

namespace {

[[nodiscard]] bool finiteConstraintCap(float x) noexcept
{
    return std::isfinite(x) && std::fabs(x) < FLT_MAX * 0.5f;
}

/// Maps one metric to [0, weight]: higher returned value is better for NEAT selection.
[[nodiscard]] float constraintScoreTerm(float v, const Constraint& c) noexcept
{
    switch (c.mode) {
    case MINIMIZE: {
        if (v <= c.target) {
            return c.weight;
        }
        if (!finiteConstraintCap(c.max_cap) || c.max_cap <= c.target) {
            return 0.f;
        }
        if (v > c.max_cap) {
            return 0.f;
        }
        return c.weight * std::clamp((c.max_cap - v) / c.max_cap, 0.f, 1.f);
    }
    case MAXIMIZE: {
        const bool has_min = finiteConstraintCap(c.min_cap) && c.min_cap > -1.e37f;
        const float lo = has_min ? c.min_cap : 0.f;
        if (has_min && v <= lo) {
            return 0.f;
        }
        const float hi = finiteConstraintCap(c.target) ? c.target : 1.f;
        if (v >= hi) {
            return c.weight;
        }
        const float span = hi - lo;
        if (span <= 1.e-6f) {
            return 0.f;
        }
        return c.weight * std::clamp((v - lo) / span, 0.f, 1.f);
    }
    case TARGET: {
        if (!finiteConstraintCap(c.min_cap) || !finiteConstraintCap(c.max_cap)) {
            return 0.f;
        }
        if (v <= c.min_cap || v >= c.max_cap) {
            return 0.f;
        }
        if (v < c.target) {
            const float span = c.target - c.min_cap;
            return span > 1.e-6f ? c.weight * (v - c.min_cap) / span : 0.f;
        }
        if (v > c.target) {
            const float span = c.max_cap - c.target;
            return span > 1.e-6f ? c.weight * (c.max_cap - v) / span : 0.f;
        }
        return c.weight;
    }
    }
    return 0.f;
}

[[nodiscard]] float sumConstraintWeights(std::initializer_list<const Constraint*> constraints) noexcept
{
    float sum = 0.f;
    for (const Constraint* c : constraints) {
        sum += c->weight;
    }
    return sum;
}

}  // namespace

void FitnessEvaluation::beginRun(
    unsigned long planned_total_ticks,
    std::uint32_t num_neurons,
    std::uint32_t num_synapses)
{
    *this = FitnessEvaluation{};
    planned_total_ticks_ = planned_total_ticks;
    num_neurons_ = num_neurons;
    num_synapses_ = num_synapses;
}

void FitnessEvaluation::updateScoreFactorsAtRuntime(
    unsigned long tick_index,
    std::uint32_t spiked_only_count,
    double tick_elapsed_ms,
    double tick_target_ms,
    bool mouse_click_edge,
    const HostFeedbackInput& feedback,
    const HostMouseOutput& mouse_out)
{
    ++ticks_recorded_;

    // behaviour trace: clicks are counted the way calculateScore judges them —
    // whichever button outvoted the other this tick — then reported as a rate
    if (mouse_out.left_click > mouse_out.right_click) {
        ++interval_left_clicks_;
    } else if (mouse_out.right_click > mouse_out.left_click) {
        ++interval_right_clicks_;
    }
    ++output_log_interval_ticks_;
    if (output_log_interval_ticks_
        >= static_cast<unsigned long>(speciation::k_output_log_tick_interval)) {
        const float inv_interval = 1.0f / static_cast<float>(output_log_interval_ticks_);
        output_log_.samples.push_back(OutputLogSample{
            mouse_out.predicted_x,
            mouse_out.predicted_y,
            static_cast<float>(interval_left_clicks_) * inv_interval,
            static_cast<float>(interval_right_clicks_) * inv_interval});
        output_log_interval_ticks_ = 0ul;
        interval_left_clicks_ = 0u;
        interval_right_clicks_ = 0u;
    }

    sum_tick_wall_ms_ += tick_elapsed_ms;
    (void)tick_target_ms;

    if (num_neurons_ > 0u) {
        const float density =
            static_cast<float>(spiked_only_count) / static_cast<float>(num_neurons_);
        sum_spike_density_ += static_cast<double>(density);
        max_spiked_only_ = std::max(max_spiked_only_, spiked_only_count);
    }

    // per-tick: accumulate prediction moments so the click can score closeness
    // to its location without buffering the history
    {
        const double px = static_cast<double>(mouse_out.predicted_x);
        const double py = static_cast<double>(mouse_out.predicted_y);
        interval_sum_x_ += px;
        interval_sum_y_ += py;
        interval_sum_xx_ += px * px;
        interval_sum_yy_ += py * py;
        ++interval_tick_count_;
    }

    if (mouse_click_edge) {
        ++click_event_count_;

        // error: squared-closeness complement of the RMS per-tick distance to
        // this click location [0,1] — moment sums make the tick average exact
        // with no history buffer, and prediction wobble raises the RMS on its
        // own, so hovering precisely at the click for the whole interval is
        // the only way to reach zero
        const double cx = static_cast<double>(mouse_out.predicted_x
            + (feedback.last_click_prediction_error_X_pos - feedback.last_click_prediction_error_X_neg));
        const double cy = static_cast<double>(mouse_out.predicted_y
            + (feedback.last_click_prediction_error_Y_pos - feedback.last_click_prediction_error_Y_neg));
        const double n = static_cast<double>(interval_tick_count_);
        const double mean_sq_dist =
            (interval_sum_xx_ - 2.0 * cx * interval_sum_x_) / n + cx * cx
            + (interval_sum_yy_ - 2.0 * cy * interval_sum_y_) / n + cy * cy;
        const double rms = std::sqrt(std::max(0.0, mean_sq_dist));
        const double closeness = 1.0 - std::min(rms, 1.0);
        const double err = 1.0 - closeness * closeness;
        interval_sum_x_ = 0.0;
        interval_sum_y_ = 0.0;
        interval_sum_xx_ = 0.0;
        interval_sum_yy_ = 0.0;
        interval_tick_count_ = 0u;

        sum_click_prediction_error_ += err;
        sum_sq_click_prediction_error_ += err * err;

        const bool left_edge = feedback.last_click_was_left_click > 0.5f;
        const bool right_edge = feedback.last_click_was_right_click > 0.5f;
        const bool pred_left = mouse_out.left_click > mouse_out.right_click;
        const bool pred_right = mouse_out.right_click > mouse_out.left_click;

        if (left_edge) {
            ++num_left_click_events_;
            if (pred_left) { ++num_left_hits_; }
        }
        if (right_edge) {
            ++num_right_click_events_;
            if (pred_right) { ++num_right_hits_; }
        }

        const bool correct_type = (left_edge && pred_left) || (right_edge && pred_right);
        sum_click_decisiveness_ +=
            std::fabs(static_cast<double>(mouse_out.right_click - mouse_out.left_click));
        ++decisiveness_samples_;

        const float certainty = mouse_out.certainty;
        const double ce = static_cast<double>(certainty);
        calib_sum_x_ += ce;
        calib_sum_y_ += err;
        calib_sum_xx_ += ce * ce;
        calib_sum_yy_ += err * err;
        calib_sum_xy_ += ce * err;
        ++calibration_samples_;
    }

    const float certainty_tick = mouse_out.certainty;
    sum_certainty_ += static_cast<double>(certainty_tick);

    prediction_history_.push_back({mouse_out.predicted_x, mouse_out.predicted_y});
    certainty_history_.push_back(certainty_tick);
    if (prediction_history_.size() > 26u) {
        prediction_history_.pop_front();
        certainty_history_.pop_front();
    }

    auto dist = [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
        const float dx = a.first - b.first;
        const float dy = a.second - b.second;
        return std::sqrt(dx * dx + dy * dy);
    };

    auto norm_jitter = [this](float d) -> float {
        return std::min(1.f, d / fitness_evaluation::k_jitter_distance_norm);
    };

    auto accumulate_weighted_jitter = [](double& sum_jitter,
                                         double& sum_weight,
                                         float jitter,
                                         float weight) {
        if (weight <= 0.0f) {
            return;
        }
        sum_jitter += static_cast<double>(weight) * static_cast<double>(jitter);
        sum_weight += static_cast<double>(weight);
    };

    const std::size_t sz = prediction_history_.size();
    if (sz >= 2u) {
        const float c_now = certainty_history_[sz - 1u];
        const float c_prev = certainty_history_[sz - 2u];
        accumulate_weighted_jitter(
            sum_soft_jitter_,
            soft_jitter_weight_,
            norm_jitter(dist(prediction_history_[sz - 1u], prediction_history_[sz - 2u])),
            std::min(c_now, c_prev));
    }
    if (sz >= 11u) {
        const float c_now = certainty_history_[sz - 1u];
        const float c_old = certainty_history_[sz - 11u];
        accumulate_weighted_jitter(
            sum_medium_jitter_,
            medium_jitter_weight_,
            norm_jitter(dist(prediction_history_[sz - 1u], prediction_history_[sz - 11u])),
            std::min(c_now, c_old));
    }
    if (sz >= 26u) {
        const float c_now = certainty_history_[sz - 1u];
        const float c_old = certainty_history_[sz - 26u];
        accumulate_weighted_jitter(
            sum_hard_jitter_,
            hard_jitter_weight_,
            norm_jitter(dist(prediction_history_[sz - 1u], prediction_history_[sz - 26u])),
            std::min(c_now, c_old));
    }

    // capture the baseline at the first tick past the 30% mark that has seen
    // a click; deferring on click-light clips keeps the learning term alive
    // instead of pinning it to a permanent zero baseline
    if (!initial_err_captured_
        && planned_total_ticks_ > 0u
        && tick_index >= (planned_total_ticks_ * 3ul) / 10ul
        && click_event_count_ > 0u) {
        initial_prediction_error_mean_ =
            static_cast<float>(sum_click_prediction_error_
                               / static_cast<double>(click_event_count_));
        initial_err_captured_ = true;
    }
}

void FitnessEvaluation::calculateScoreFactors(
    std::uint32_t inhibitory_spike_rate,
    std::uint32_t excitatory_spike_rate)
{
    score_factors_ = ScoreFactors{};
    if (ticks_recorded_ == 0u) {
        return;
    }

    const double n = static_cast<double>(ticks_recorded_);
    score_factors_.milliseconds_per_tick =
        static_cast<float>(sum_tick_wall_ms_ / n);
    score_factors_.mean_spike_density =
        num_neurons_ > 0u ? static_cast<float>(sum_spike_density_ / n) : 0.f;
    score_factors_.max_spike_density =
        num_neurons_ > 0u
            ? static_cast<float>(static_cast<double>(max_spiked_only_)
                                   / static_cast<double>(num_neurons_))
            : 0.f;
    const double w = static_cast<double>(inhibitory_spike_rate)
        + static_cast<double>(excitatory_spike_rate);
    score_factors_.inhibitory_excitatory_ratio =
        w > 0.0
            ? static_cast<float>(
                  static_cast<double>(inhibitory_spike_rate) / w)
            : 0.f;
    score_factors_.snn_neuron_synapse_ratio =
        num_neurons_ > 0u
            ? static_cast<float>(
                  static_cast<double>(num_synapses_)
                  / static_cast<double>(num_neurons_))
            : 0.f;

    if (click_event_count_ > 0u) {
        const double mean_click_err =
            sum_click_prediction_error_ / static_cast<double>(click_event_count_);
        const double ex2 =
            sum_sq_click_prediction_error_
            / static_cast<double>(click_event_count_);
        const double var = std::max(0.0, ex2 - mean_click_err * mean_click_err);
        const double std_err = std::sqrt(var);
        score_factors_.mean_last_click_prediction_error =
            static_cast<float>(mean_click_err);
        score_factors_.consistency_last_click_prediction_error = static_cast<float>(
            std::clamp(1.0 - 2.0 * std_err, 0.0, 1.0));
    }

    score_factors_.mean_last_click_left_hits =
        num_left_click_events_ > 0u
            ? static_cast<float>(
                  static_cast<double>(num_left_hits_)
                  / static_cast<double>(num_left_click_events_))
            : 0.f;
    score_factors_.mean_last_click_right_hits =
        num_right_click_events_ > 0u
            ? static_cast<float>(
                  static_cast<double>(num_right_hits_)
                  / static_cast<double>(num_right_click_events_))
            : 0.f;

    score_factors_.mean_click_type_decisiveness =
        decisiveness_samples_ > 0u
            ? static_cast<float>(
                  sum_click_decisiveness_
                  / static_cast<double>(decisiveness_samples_))
            : 0.f;

    score_factors_.mean_certainty =
        static_cast<float>(sum_certainty_ / n);

    if (calibration_samples_ >= 2u) {
        const double dn = static_cast<double>(calibration_samples_);
        const double num = dn * calib_sum_xy_ - calib_sum_x_ * calib_sum_y_;
        const double den_x = dn * calib_sum_xx_ - calib_sum_x_ * calib_sum_x_;
        const double den_y = dn * calib_sum_yy_ - calib_sum_y_ * calib_sum_y_;
        if (den_x > 1e-20 && den_y > 1e-20) {
            double r = num / std::sqrt(den_x * den_y);
            r = std::clamp(r, -1.0, 1.0);
            score_factors_.calibration_error =
                static_cast<float>((r + 1.0) * 0.5);
        }
    }

    score_factors_.soft_jitter =
        soft_jitter_weight_ > 0.0
            ? static_cast<float>(sum_soft_jitter_ / soft_jitter_weight_)
            : 0.f;
    score_factors_.medium_jitter =
        medium_jitter_weight_ > 0.0
            ? static_cast<float>(sum_medium_jitter_ / medium_jitter_weight_)
            : 0.f;
    score_factors_.hard_jitter =
        hard_jitter_weight_ > 0.0
            ? static_cast<float>(sum_hard_jitter_ / hard_jitter_weight_)
            : 0.f;

    if (initial_err_captured_ && click_event_count_ > 0u) {
        const float final_click_mean = static_cast<float>(
            sum_click_prediction_error_
            / static_cast<double>(click_event_count_));
        const float denom = std::max(initial_prediction_error_mean_, 1e-6f);
        const float improve =
            (initial_prediction_error_mean_ - final_click_mean) / denom;
        score_factors_.learning = std::clamp(improve, 0.f, 1.f);
    } else {
        score_factors_.learning = 0.f;
    }
}

int FitnessEvaluation::calculateScore()
{
    if (ticks_recorded_ == 0u) {
        Util::Logs::write("no ticks recorded");
        return 0;
    }

    {
        std::ostringstream line;
        line << "ticks recorded: " << ticks_recorded_
             << " click events: " << click_event_count_;
        Util::Logs::write(line.str());
    }

    float total = 0.f;

    // Weights follow priority: 
    // (1) accuracy 
    // (2) consistency
    // (3) certainty vs accuracy
    // (4) certainty 
    // (5) speed 
    // (6) no hiccups 
    // (7) energy/spikes 
    // (8) learning
    // (9) prediction stability jitter
    // Unlisted metrics are structural regularizers
    //
    // Score term = weight * quality in [0,1] per `constraintScoreTerm` (higher is better).

    // --- (5) speed ---
    const Constraint c_milliseconds_per_tick = {2.0f, -FLT_MAX, -FLT_MAX, 16.0f, MINIMIZE};

    // --- (7) spikes / energy (soft nudge only; must not override task terms) ---
    const Constraint c_mean_spike_density = {1.5f, fitness_evaluation::k_target_mean_spike_density, 0.015f, 0.19f, TARGET};
    const Constraint c_max_spike_density = {1.5f, 0.12f, 0.02f, 0.48f, TARGET};

    // --- mild topology / dynamics priors (inferred: stability, not goal-critical) ---
    const Constraint c_inhibitory_excitatory_ratio = {0.8f, 0.20f, 0.08f, 0.38f, TARGET};
    const Constraint c_snn_neuron_synapse_ratio = {1.5f, 95.0f, 6.0f, 240.0f, TARGET};

    // --- (1) click location + button accuracy: error is the squared-closeness
    //     complement of the RMS per-tick prediction distance to each click since
    //     the previous one, so it rewards hovering at the target early and
    //     precisely, not just being there at the click tick ---
    const Constraint c_mean_prediction_error = {24.0f, -FLT_MAX, -FLT_MAX, 1.0f, MINIMIZE};
    const Constraint c_left_hits = {14.f, FLT_MAX, 0.00f, FLT_MAX, MAXIMIZE};
    const Constraint c_right_hits = {14.f, FLT_MAX, 0.00f, FLT_MAX, MAXIMIZE};
    const Constraint c_click_decisiveness = {5.0f, FLT_MAX, 0.00f, FLT_MAX, MAXIMIZE};

    // --- (2) consistency (quality score in [0,1], higher = steadier; scored gated by closeness below) ---
    const Constraint c_consistency_quality = {8.0f, FLT_MAX, 0.00f, FLT_MAX, MAXIMIZE};

    // --- (3) certainty vs accuracy: mapped Pearson score, lower is better-calibrated ---
    const Constraint c_calibration_score = {1.5f, -FLT_MAX, -FLT_MAX, 1.0f, MINIMIZE};

    // --- (4) mean certainty: TARGET band rewards moderate vote concentration so
    //     multi-modal bins and parabolic sub-bin interpolation are not punished;
    //     calibration term already handles "confidence should mean something" ---
    const Constraint c_certainty = {2.0f, 0.67f, 0.0f, 1.0f, TARGET};

    // --- (8) learning ---
    const Constraint c_learning = {3.5f, FLT_MAX, 0.0f, FLT_MAX, MAXIMIZE};

    // --- (9) SNN predicted position stability (lower is better) ---
    const Constraint c_soft_jitter = {0.6f, -FLT_MAX, -FLT_MAX, 1.0f, MINIMIZE};
    const Constraint c_medium_jitter = {0.9f, -FLT_MAX, -FLT_MAX, 1.0f, MINIMIZE};
    const Constraint c_hard_jitter = {1.1f, -FLT_MAX, -FLT_MAX, 1.0f, MINIMIZE};

    total += constraintScoreTerm(score_factors_.milliseconds_per_tick, c_milliseconds_per_tick);

    total += constraintScoreTerm(score_factors_.mean_spike_density, c_mean_spike_density);
    total += constraintScoreTerm(score_factors_.max_spike_density, c_max_spike_density);

    total += constraintScoreTerm(score_factors_.inhibitory_excitatory_ratio, c_inhibitory_excitatory_ratio);
    total += constraintScoreTerm(score_factors_.snn_neuron_synapse_ratio, c_snn_neuron_synapse_ratio);

    const bool had_clicks = click_event_count_ > 0u;
    const float prediction_error_for_score = had_clicks
        ? score_factors_.mean_last_click_prediction_error
        : c_mean_prediction_error.max_cap;
    // consistency gated by achieved closeness so steadiness only pays in
    // proportion to actually being near targets — consistent mediocrity
    // (statue parked at the click centroid) no longer collects the full bonus
    const float consistency_for_score = had_clicks
        ? score_factors_.consistency_last_click_prediction_error
              * (1.f - score_factors_.mean_last_click_prediction_error)
        : 0.f;
    const float learning_for_score = had_clicks ? score_factors_.learning : 0.f;

    total += constraintScoreTerm(prediction_error_for_score, c_mean_prediction_error);
    if (num_left_click_events_ > 0u)
        total += constraintScoreTerm(score_factors_.mean_last_click_left_hits, c_left_hits);
    if (num_right_click_events_ > 0u)
        total += constraintScoreTerm(score_factors_.mean_last_click_right_hits, c_right_hits);
    total += constraintScoreTerm(score_factors_.mean_click_type_decisiveness, c_click_decisiveness);

    total += constraintScoreTerm(consistency_for_score, c_consistency_quality);
    if (had_clicks && calibration_samples_ >= 2u)
        total += constraintScoreTerm(score_factors_.calibration_error, c_calibration_score);
    total += constraintScoreTerm(score_factors_.mean_certainty, c_certainty);

    total += constraintScoreTerm(learning_for_score, c_learning);

    total += constraintScoreTerm(score_factors_.soft_jitter, c_soft_jitter);
    total += constraintScoreTerm(score_factors_.medium_jitter, c_medium_jitter);
    total += constraintScoreTerm(score_factors_.hard_jitter, c_hard_jitter);

    const float max_raw_fitness = sumConstraintWeights({
        &c_milliseconds_per_tick,
        &c_mean_spike_density,
        &c_max_spike_density,
        &c_inhibitory_excitatory_ratio,
        &c_snn_neuron_synapse_ratio,
        &c_mean_prediction_error,
        &c_left_hits,
        &c_right_hits,
        &c_click_decisiveness,
        &c_consistency_quality,
        &c_calibration_score,
        &c_certainty,
        &c_learning,
        &c_soft_jitter,
        &c_medium_jitter,
        &c_hard_jitter,
    });

    {
        std::ostringstream line;
        line << "total fitness score: " << std::fixed << std::setprecision(5) << total << '/'
             << max_raw_fitness << std::defaultfloat << std::setprecision(6);
        Util::Logs::write(line.str());
    }
    return static_cast<int>(std::lround(total * 100.f));
    
}

