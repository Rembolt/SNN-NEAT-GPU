#pragma once
/*
ScoreFactors (see Projects Ideas.txt): spike density, tick compute time, I/(I+E),
click prediction / calibration / jitter / reaction / learning — all normalized per doc.
*/
#include "DNAviews.h"
#include "HostFeedbackInput.h"
#include "HostMouseOutput.h"

#include <cstdint>
#include <deque>
#include <utility>



enum ConstraintMode { MINIMIZE, MAXIMIZE, TARGET };

struct Constraint {
    float weight;
    float target; ///< TARGET mode only: peak value. Unused (set 0) for MINIMIZE/MAXIMIZE — those use hardcoded bounds (0 and 1)
    float min_cap; ///< MAXIMIZE: floor for ramp; TARGET: lower band edge (outside → 0 credit)
    float max_cap; ///< MINIMIZE: value at 0 credit; MAXIMIZE: unused if +inf; TARGET: upper band edge
    ConstraintMode mode;
};

class FitnessEvaluation {
public:
    FitnessEvaluation() = default;

    void beginRun(
        unsigned long planned_total_ticks,
        std::uint32_t num_neurons,
        std::uint32_t num_synapses);

    /// Call once per simulated tick after kernels + ErrorFeedback for that tick are up to date.
    /// \param tick_elapsed_ms wall time for this tick's work (excludes post-tick scheduler sleep).
    /// \param tick_target_ms configured per-tick budget in milliseconds.
    void updateScoreFactorsAtRuntime(
        unsigned long tick_index,
        std::uint32_t spiked_only_count,
        double tick_elapsed_ms,
        double tick_target_ms,
        bool mouse_click_edge,
        const HostFeedbackInput& feedback,
        const HostMouseOutput& mouse_out);

    void calculateScoreFactors(
        std::uint32_t inhibitory_spike_rate,
        std::uint32_t excitatory_spike_rate);

    int calculateScore();

    [[nodiscard]] const ScoreFactors& scoreFactors() const noexcept { return score_factors_; }

    [[nodiscard]] const OutputLog& outputLog() const noexcept { return output_log_; }

private:

    ScoreFactors score_factors_{};

    /// behaviour trace handed to speciation
    OutputLog output_log_{};
    unsigned long output_log_interval_ticks_{};
    std::uint32_t interval_left_clicks_{};
    std::uint32_t interval_right_clicks_{};

    unsigned long planned_total_ticks_{};
    std::uint32_t num_neurons_{};
    std::uint32_t num_synapses_{};

    unsigned long ticks_recorded_{};

    double sum_tick_wall_ms_{};

    double sum_spike_density_{};
    std::uint32_t max_spiked_only_{};

    /// Sum of squared click prediction errors (same samples as sum_click_prediction_error_).
    double sum_sq_click_prediction_error_{};

    std::uint32_t num_left_hits_{};
    std::uint32_t num_right_hits_{};
    std::uint32_t num_left_click_events_{};
    std::uint32_t num_right_click_events_{};
    double sum_click_decisiveness_{};
    unsigned long decisiveness_samples_{};

    double sum_certainty_{};

    double calib_sum_x_{};
    double calib_sum_y_{};
    double calib_sum_xx_{};
    double calib_sum_yy_{};
    double calib_sum_xy_{};
    unsigned long calibration_samples_{};

    double sum_soft_jitter_{};
    double soft_jitter_weight_{};
    double sum_medium_jitter_{};
    double medium_jitter_weight_{};
    double sum_hard_jitter_{};
    double hard_jitter_weight_{};

    std::deque<std::pair<float, float>> prediction_history_{};
    std::deque<float> certainty_history_{};

    /// Moment sums of predicted positions since the previous click — lets the
    /// click compute RMS distance to its location without a per-tick buffer
    double interval_sum_x_{};
    double interval_sum_y_{};
    double interval_sum_xx_{};
    double interval_sum_yy_{};
    unsigned long interval_tick_count_{};

    unsigned long click_event_count_{};
    double sum_click_prediction_error_{}; ///< Sum of last_click_prediction_error on physical click edges only.
    bool initial_err_captured_{false};
    float initial_prediction_error_mean_{};
};
