#pragma once

#include "HostFeedbackInput.h"
#include "HostMouseInput.h"
#include "util/Util.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace debug {

#if defined(SNN_GPU_DEBUG_LOGS)
inline constexpr bool k_debug_logs = true;
#else
inline constexpr bool k_debug_logs = false;
#endif

}  // namespace debug

namespace runtime {

inline constexpr auto k_tick_target = std::chrono::milliseconds(16);

}  // namespace runtime

namespace dna_layout {

constexpr uint32_t k_neuron_kind_hidden = 0u;
constexpr uint32_t k_neuron_kind_input = 1u;
constexpr uint32_t k_neuron_kind_output = 2u;
constexpr uint32_t k_neuron_kind_bias = 3u;
constexpr uint32_t k_no_kind_target = 0u;

constexpr uint32_t k_mouse_output_x_bins = 48u;
constexpr uint32_t k_mouse_output_y_bins = 48u;
constexpr uint32_t k_mouse_output_click_slots = 2u;  // right click + left click
constexpr uint32_t k_output_slot_count =
    k_mouse_output_x_bins + k_mouse_output_y_bins + k_mouse_output_click_slots;

constexpr int k_input_slot_count = static_cast<int>(
    Util::hostInputStructFloatSlotCount<HostMouseInput>()
    + Util::hostInputStructFloatSlotCount<HostFeedbackInput>());

}  // namespace dna_layout

// Packed d_input indices shared by host layout and STDP kernel args
namespace kernel_input_layout {

inline constexpr std::size_t k_mouse_input_float_count =
    Util::hostInputStructFloatSlotCount<HostMouseInput>();
inline constexpr std::size_t k_feedback_temporal_decay_float_index =
    offsetof(HostFeedbackInput, last_click_prediction_temporal_decay) / sizeof(float);
inline constexpr std::size_t k_feedback_last_click_distance_float_index =
    offsetof(HostFeedbackInput, last_click_distance) / sizeof(float);
inline constexpr std::size_t k_temporal_decay_input =
    k_mouse_input_float_count + k_feedback_temporal_decay_float_index;
inline constexpr std::size_t k_last_click_distance_input =
    k_mouse_input_float_count + k_feedback_last_click_distance_float_index;

}  // namespace kernel_input_layout

namespace neat_pipeline {

constexpr unsigned long k_train_tick_count = 2813ul;  // (50 for testing) later change to 2,813 ticks ~ 45 seconds
constexpr int k_clicks_required = 10;  // (1 for testing) 9 for training
constexpr int k_record_attempts = 2;
constexpr double k_percentage_of_organism_to_test_more_times = 0.13;
constexpr std::size_t k_progress_test_top_organisms = 3;
constexpr std::size_t k_progress_test_max_clips = 5;

}  // namespace neat_pipeline

namespace live_organism {

constexpr unsigned long k_live_tick_count = 3000ul;

}  // namespace live_organism

namespace mouse_clip {

inline constexpr char k_dir_name[] = "mouse-input-log-clips";
inline constexpr char k_progress_test_dir_name[] = "progress-test";
inline constexpr char k_extension[] = ".miclip";
constexpr std::uint32_t k_magic = 0x504C434Du;  // 'MICP' little-endian
constexpr std::uint32_t k_version = 1u;
constexpr int k_target_clip_count = 10;

}  // namespace mouse_clip

namespace fitness_evaluation {

constexpr float k_stimulus_speed = 0.05f;
constexpr float k_direction_dot_threshold = 0.5f;
constexpr float k_response_prediction_motion = 0.02f;
constexpr float k_jitter_distance_norm = 1.4142135623730950488f;  // sqrt(2), max unit-square step
constexpr float k_target_mean_spike_density = 0.07f;

}  // namespace fitness_evaluation

namespace mouse_output_ui {

constexpr float k_ghost_cursor_radius = 10.0f;
constexpr int k_overlay_size_px = 40;

}  // namespace mouse_output_ui

namespace test_harness {

constexpr std::size_t k_click_distance_accuracy_input =
    kernel_input_layout::k_last_click_distance_input;
constexpr std::size_t k_temporal_decay_input =
    kernel_input_layout::k_temporal_decay_input;
constexpr float k_changed_epsilon = 1.0e-5f;

}  // namespace test_harness

namespace speciation {

    // behaviour distance is already normalized to [0,1], so the threshold reads as a
    // percentage: two organisms are kin while their traces differ by less than 15%
    constexpr float k_compatibility_threshold = 0.15f;
    // behaviour sample every N ticks; 25 keeps ~112 samples over a train run
    constexpr int k_output_log_tick_interval = 25;
    // where the mouse was predicted matters more than which button
    constexpr float k_behavior_position_weight = 0.7f;
    constexpr float k_behavior_click_weight = 0.3f;
    // genome-structure comparison is no longer used for speciation
    //constexpr float k_comparison_excess_coeff = 1.0f;
    //constexpr float k_comparison_disjoint_coeff = 1.0f;
    //constexpr float k_comparison_param_coeff = 1.5f;
    //constexpr int k_comparison_min_genome_size = 20;
    constexpr int k_target_min_species = 4;
    constexpr int k_target_max_species = 9;
    // behaviour clusters drift fast, so nudge the threshold by fractions of a percent
    // and let a few generations settle the species count inside the lenient 4-9 band
    constexpr float k_threshold_raise_step = 0.005f;
    constexpr float k_threshold_lower_step = 0.004f;
    constexpr float k_threshold_absolute_min = 0.02f;
    constexpr float k_threshold_absolute_max = 0.60f;
    constexpr int k_stagnation_limit = 100;

}  // namespace speciation
namespace mutation {
    // incremental perturb = gaussian(current, sigma), sigma = uniform(mod_min,mod_max) * (param_max - param_min)
    constexpr float k_decimal_mutation_mod_min = 0.010f;
    constexpr float k_decimal_mutation_mod_max = 0.025f;

    constexpr int k_integer_mutation_mod_min = 1;
    constexpr int k_integer_mutation_mod_max = 2;  // not 3; delay jumps are disruptive
    // one global block per organism — keep rare
    constexpr float k_global_genome_mutation_chance = 0.07f;
    constexpr float k_global_genome_random_change_chance = 0.03f;

    constexpr float k_hidden_neuron_mutation_chance = 0.12f;
    constexpr float k_hidden_neuron_random_change_chance = 0.04f;

    constexpr float k_output_neuron_mutation_chance = 0.06f;
    constexpr float k_output_neuron_random_change_chance = 0.02f;

    constexpr float k_bias_neuron_mutation_chance = 0.f;
    constexpr float k_bias_neuron_random_change_chance = 0.f;

    constexpr float k_input_neuron_mutation_chance = 0.f;  // or 0
    constexpr float k_input_neuron_random_change_chance = 0.f;

    constexpr float k_connection_genome_mutation_chance = 0.07f;
    constexpr float k_connection_genome_random_change_chance = 0.01f;
    // per child, not per gene
    constexpr float k_new_connection_genome_chance = 0.07f;
    constexpr float k_disable_connection_genome_chance = 0.03f;
    constexpr float k_enable_connection_genome_chance = 0.025f;
    constexpr float k_new_neuron_genome_chance = 0.017f;
}

namespace organism_ranges {

inline std::pair<float, float> R(std::pair<int, int> r){return{float(r.first), float(r.second)};}

// Per-parameter mutation and distance bounds.
// preferred_*: biologically plausible range; used for random init, sigma, and distance normalization
// absolute_*:  hard clamp applied after mutation (references inline range vars below — single source of truth)
// mutation_scale_pct: sigma = pct * (preferred_max - preferred_min); 0 = use integer step (ignored for integers)
struct ParamBounds {
    float preferred_min;
    float preferred_max;
    float absolute_min;
    float absolute_max;
    float mutation_scale_pct;
};

// global_genome
inline const std::pair<int, int> species_id{0, 100};
inline const std::pair<int, int> organism_id{0, 10'000'000};
inline const std::pair<int, int> generation{0, 1'000'000};
inline const std::pair<float, float> fitness{0.0f, 1.0e6f};
inline const std::pair<float, float> adjusted_fitness{0.0f, 1.0e6f};
inline const std::pair<float, float> membrane_capacitance{1.0f, 1000.0f};
inline const std::pair<float, float> leak_conductance{0.1f, 100.0f};
inline const std::pair<float, float> leak_reversal_potential{-90.0f, -40.0f};
inline const std::pair<float, float> slope_factor{0.1f, 15.0f};
inline const std::pair<float, float> spike_peak_voltage{0.0f, 50.0f};
inline const std::pair<float, float> homeostasis_weight_mod{0.0f, 1.0f};
inline const std::pair<float, float> homeostasis_pot_bias_mod{0.0f, 1.0f};
inline const std::pair<float, float> homeostasis_dep_bias_mod{0.0f, 1.0f};
inline const std::pair<float, float> homeostasis_threshold_mod{0.0f, 1.0f};
inline const std::pair<float, float> homeostasis_rate_max_distance{0.0f, 100.0f};
inline const std::pair<int, int> ticks_between_homeostasis{1, 10'000};
inline const std::pair<int, int> max_trace_tick{1, 1'000};
inline const std::pair<int, int> max_delay{1, 100};
inline const std::pair<float, float> click_distance_weight{0.0f, 1.0f};
inline const std::pair<float, float> output_axis_decay{0.0f, 1.0f};
inline const std::pair<int, int> feedback_temporal_decay_ticks{1, 10'000};

// neuron_genomes
inline const std::pair<int, int> innovation_id{0, std::numeric_limits<int>::max()};
//inline const std::pair<int, int> split_connection_id{0, std::numeric_limits<int>::max()};
inline const std::vector<std::string> neuron_type{"input", "hidden", "output", "bias"};
//inline const std::pair<int, int> layer_depth{0, 32};
inline const std::pair<float, float> a_subthreshold_adaptation{-10.0f, 20.0f};
inline const std::pair<float, float> b_spike_triggered_adaptation{0.0f, 400.0f};
inline const std::pair<float, float> tau_w_adaptation_time_constant{1.0f, 1000.0f};
inline const std::pair<float, float> v_threshold{-70.0f, -30.0f};
inline const std::pair<float, float> v_rest{-90.0f, -30.0f};
inline const std::pair<int, int> refractory_length_ticks{1, 100};
inline const std::pair<float, float> target_spike_rate{0.0f, 200.0f};
// input: [0, k_input_slot_count); output: [0, k_output_slot_count); hidden/bias: 0
inline const std::pair<int, int> neuron_type_target_input{0, dna_layout::k_input_slot_count - 1};
inline const std::pair<int, int> neuron_type_target_output{
    0, static_cast<int>(dna_layout::k_output_slot_count) - 1};
inline const std::pair<int, int> neuron_type_target_hidden_bias{0, 0};

// connection_genomes
inline const std::pair<int, int> enabled{0, 1};
inline const std::pair<float, float> initial_weight{-100.0f, 100.0f};
inline const std::pair<int, int> delay_ticks{1, max_delay.second};
inline const std::pair<float, float> potentiation_bias{0.0f, 1.0f};
inline const std::pair<float, float> depression_bias{0.0f, 1.0f};

// AdEx biological units (mV, ms, nS, pF, pA) + other wide-range params
// absolute_min/max reference the inline range vars above — no duplicate literals
inline const std::unordered_map<std::string, ParamBounds> param_bounds{
    {"leak_reversal_potential",        {-75.0f, -50.0f,  leak_reversal_potential.first,                              leak_reversal_potential.second,                           0.02f}},
    {"v_threshold",                    {-55.0f, -40.0f,  v_threshold.first,                                          v_threshold.second,                                       0.02f}},
    {"v_rest",                         {-70.0f, -45.0f,  v_rest.first,                                               v_rest.second,                                            0.03f}},
    {"slope_factor",                   {  1.0f,   5.0f,  slope_factor.first,                                         slope_factor.second,                                      0.05f}},
    {"a_subthreshold_adaptation",      { -1.0f,   5.0f,  a_subthreshold_adaptation.first,                            a_subthreshold_adaptation.second,                         0.05f}},
    {"b_spike_triggered_adaptation",   {  0.0f, 100.0f,  b_spike_triggered_adaptation.first,                         b_spike_triggered_adaptation.second,                      0.08f}},
    {"tau_w_adaptation_time_constant", { 30.0f, 300.0f,  tau_w_adaptation_time_constant.first,                       tau_w_adaptation_time_constant.second,                    0.10f}},
    {"membrane_capacitance",           {100.0f, 300.0f,  membrane_capacitance.first,                                 membrane_capacitance.second,                              0.05f}},
    {"leak_conductance",               { 10.0f,  30.0f,  leak_conductance.first,                                     leak_conductance.second,                                  0.05f}},
    {"spike_peak_voltage",             { 15.0f,  40.0f,  spike_peak_voltage.first,                                   spike_peak_voltage.second,                                0.03f}},
    {"homeostasis_rate_max_distance",  {  1.5f,  10.0f,  homeostasis_rate_max_distance.first,                        homeostasis_rate_max_distance.second,                     0.05f}},
    // homeo blend gains [0,1]: preferred wide vs legend clusters (weight~0.1-0.23, pot~0.04-0.12, dep bimodal~0.12-0.75, thresh~0.28-0.5)
    {"homeostasis_weight_mod",         {  0.02f,  0.55f, homeostasis_weight_mod.first,                               homeostasis_weight_mod.second,                            0.08f}},
    {"homeostasis_pot_bias_mod",       {  0.01f,  0.45f, homeostasis_pot_bias_mod.first,                             homeostasis_pot_bias_mod.second,                          0.08f}},
    {"homeostasis_dep_bias_mod",       {  0.02f,  0.85f, homeostasis_dep_bias_mod.first,                             homeostasis_dep_bias_mod.second,                          0.08f}},
    {"homeostasis_threshold_mod",      {  0.10f,  0.70f, homeostasis_threshold_mod.first,                            homeostasis_threshold_mod.second,                         0.08f}},
    {"target_spike_rate",              {  0.0f, 100.0f,  target_spike_rate.first,                                    target_spike_rate.second,                                 0.05f}},
    {"initial_weight",                 {-10.0f,  10.0f,  initial_weight.first,                                       initial_weight.second,                                    0.08f}},
    // STDP synapse biases: legends span nearly full [0,1]; keep preferred wide, avoid exact 0/1 on random reset
    {"potentiation_bias",              {  0.05f,  0.95f, potentiation_bias.first,                                    potentiation_bias.second,                                 0.08f}},
    {"depression_bias",                {  0.05f,  0.95f, depression_bias.first,                                      depression_bias.second,                                   0.08f}},
    // integer params: preferred range used for random init + distance normalization (mutation_scale_pct unused)
    {"ticks_between_homeostasis",      { 3.0f, 2000.0f, static_cast<float>(ticks_between_homeostasis.first),        static_cast<float>(ticks_between_homeostasis.second),     0.0f}},
    {"max_trace_tick",                 { 10.0f,  300.0f, static_cast<float>(max_trace_tick.first),                   static_cast<float>(max_trace_tick.second),                0.0f}},
    {"max_delay",                      {  1.0f,   100.0f, static_cast<float>(max_delay.first),                        static_cast<float>(max_delay.second),                     0.0f}},
    {"feedback_temporal_decay_ticks",  { 60.0f, 1200.0f, static_cast<float>(feedback_temporal_decay_ticks.first),    static_cast<float>(feedback_temporal_decay_ticks.second), 0.0f}},
    {"delay_ticks",                    {  1.0f,   90.0f, static_cast<float>(delay_ticks.first),                      static_cast<float>(delay_ticks.second),                   0.0f}},
    {"refractory_length_ticks",        {  1.0f,   20.0f, static_cast<float>(refractory_length_ticks.first),          static_cast<float>(refractory_length_ticks.second),       0.0f}},
};

inline const std::unordered_map<std::string, std::pair<float, float>> genome_ranges{
    {"species_id", R(species_id)}, 
    {"organism_id", R(organism_id)}, 
    {"generation", R(generation)},
    {"fitness", fitness}, 
    {"adjusted_fitness", adjusted_fitness},
    {"membrane_capacitance", membrane_capacitance}, 
    {"leak_conductance", leak_conductance},
    {"leak_reversal_potential", leak_reversal_potential}, 
    {"slope_factor", slope_factor},
    {"spike_peak_voltage", spike_peak_voltage}, 
    {"homeostasis_weight_mod", homeostasis_weight_mod},
    {"homeostasis_pot_bias_mod", homeostasis_pot_bias_mod},
    {"homeostasis_dep_bias_mod", homeostasis_dep_bias_mod},
    {"homeostasis_threshold_mod", homeostasis_threshold_mod},
    {"homeostasis_rate_max_distance", homeostasis_rate_max_distance},
    {"ticks_between_homeostasis", R(ticks_between_homeostasis)}, 
    {"max_trace_tick", R(max_trace_tick)},
    {"max_delay", R(max_delay)}, 
    {"click_distance_weight", click_distance_weight},
    {"output_axis_decay", output_axis_decay},
    {"feedback_temporal_decay_ticks", R(feedback_temporal_decay_ticks)},
    {"innovation_id", R(innovation_id)}, 
    {"a_subthreshold_adaptation", a_subthreshold_adaptation},
    {"b_spike_triggered_adaptation", b_spike_triggered_adaptation},
    {"tau_w_adaptation_time_constant", tau_w_adaptation_time_constant}, 
    {"v_threshold", v_threshold},
    {"v_rest", v_rest}, 
    {"refractory_length_ticks", R(refractory_length_ticks)},
    {"target_spike_rate", target_spike_rate},
    {"enabled", R(enabled)},
    {"initial_weight", initial_weight}, 
    {"delay_ticks", R(delay_ticks)},
    {"potentiation_bias", potentiation_bias}, 
    {"depression_bias", depression_bias},
};

}  // namespace organism_ranges
