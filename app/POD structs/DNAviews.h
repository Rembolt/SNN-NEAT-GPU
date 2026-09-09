
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ScoreFactors {
    float milliseconds_per_tick{}; ///< How fast tick() finished; scheduler wait does not count [ms]
    float mean_spike_density{}; ///< 0–1 fraction of neurons that fire per tick on average [0,1]
    float max_spike_density {}; ///< Max 0–1 fraction of neurons firing in any single tick [0,1]
    float inhibitory_excitatory_ratio{}; ///< I / (I + E): inhibitory vs excitatory spike delivery over the run [0,1]
    float snn_neuron_synapse_ratio{}; ///< Synapses per neuron (num_synapses / num_neurons); e.g. 10 when average out-degree is 10
    float mean_last_click_prediction_error{}; ///< Mean per-click squared-closeness complement of RMS per-tick prediction distance to the click [0,1]
    float consistency_last_click_prediction_error{}; ///< Variability of prediction precision [0 = random, 1 = equally precise]
    float mean_last_click_left_hits{}; ///< Fraction of left clicks where next-click type was predicted correctly [0,1]
    float mean_last_click_right_hits{}; ///< Fraction of right clicks where next-click type was predicted correctly [0,1]
    float mean_click_type_decisiveness{}; ///< When click type was correct, average |right_click − left_click| [0,1]
    float mean_certainty{}; ///< Mean certainty of predicted next click location [0,1]
    float calibration_error{}; ///< Correlation between certainty and last_click_prediction_error, mapped to [0,1]
    float soft_jitter{}; ///< Frame-to-frame predicted_x/y movement, certainty-weighted [0,1]
    float medium_jitter{}; ///< Predicted position drift vs 10 ticks ago, certainty-weighted [0,1]
    float hard_jitter{}; ///< Predicted position drift vs 25 ticks ago, certainty-weighted [0,1]
    float learning{}; ///< Improvement in click-location prediction from ~30% of run to end [0,1]
};

struct Specie {

    int32_t id;
    std::string rep;
    int32_t rep_id;
    float rep_fitness;
    int32_t created_generation;
    int32_t size_this_gen;
    float best_fitness_this_gen;
    float best_fitness_ever;
    int32_t stagnation_generations;
    int offspring_count;
    ScoreFactors score_factors;

};

/// One behaviour sample taken every speciation::k_output_log_tick_interval ticks
struct OutputLogSample {
    float predicted_x{}; ///< [0,1] NORMALIZED
    float predicted_y{}; ///< [0,1] NORMALIZED
    float left_click_rate{}; ///< Ticks in the interval where left outvoted right, over interval length [0,1]
    float right_click_rate{}; ///< Ticks in the interval where right outvoted left, over interval length [0,1]
};

/// Behaviour trace of a run on one clip: what the organism did, not what it is.
/// Lives in memory only — every generation runs a new clip, so traces from
/// different clips are not comparable and are never persisted
struct OutputLog {
    std::vector<OutputLogSample> samples;
};

struct OrganismLog {
    std::string organism_path;
    int organism_id;
    int generation;
    int specie_id;
    float raw_fitness;
    float adjusted_fitness;
    ScoreFactors score_factors;
    OutputLog output_log;
};