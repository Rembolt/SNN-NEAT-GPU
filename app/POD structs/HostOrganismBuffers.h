#pragma once
 
#include <cstdint>
#include <vector>
 
// Host-side constants and seed values for one organism.
struct HostGlobalParams {
    

    uint32_t h_current_tick;
    uint32_t h_words_per_row;
    uint32_t h_num_of_input_neurons;
    uint32_t h_num_output_slots;
    float    h_trace_tau;
    uint32_t h_num_neurons;
    uint32_t h_num_synapses;
    uint32_t h_max_trace_len;
    uint32_t h_max_delay;
    float    h_click_distance_weight;
    float    h_homeo_weight_mod;
    float    h_homeo_pot_mod;
    float    h_homeo_dep_mod;
    float    h_homeo_thresh_mod;
    float    h_homeo_rate_max_distance;
    uint32_t h_ticks_between_homeostasis;
    float    h_capacitance_C;
    float    h_leak_g;
    float    h_leak_E;
    float    h_slope_delta_T;
    float    h_v_peak;
    float    h_dt_ms;
    uint32_t h_inhibitory_spike_rate;
    uint32_t h_excitatory_spike_rate;
    std::vector<uint32_t> h_output_slot_neuron_counts;
    uint32_t h_output_size;
};
 
struct HostNeuronBuffers {
    std::vector<float>    h_V;
    std::vector<float>    h_W;
    std::vector<float>    h_threshold_V;
    std::vector<uint32_t> h_refract_end;
    std::vector<uint32_t> h_refractory_length;
    std::vector<uint32_t> h_last_delievery_update;
    std::vector<uint32_t> h_last_spike;
    std::vector<float>    h_spike_rate; // leaky firing-rate trace; decays by exp(-1/fire_rate_tau) each tick
    std::vector<float>    h_spike_rate_target;
    std::vector<float>    h_a;
    std::vector<float>    h_b;
    std::vector<float>    h_tau_w;
    std::vector<float>    h_v_rest;
    std::vector<uint32_t> h_neuron_kind;
    std::vector<uint32_t> h_neuron_kind_target;

};
 
struct HostCSRCSCBuffers {
    std::vector<uint32_t> h_csr_row_ptr;
    std::vector<uint32_t> h_csr_col_idx;
    std::vector<uint32_t> h_csc_row_ptr;
    std::vector<uint32_t> h_csc_col_idx;
    std::vector<uint32_t> h_csc_to_csr_index_mapping;
    std::vector<float>    h_weight;
    std::vector<float>    h_delay;
    std::vector<float>    h_potentiation_bias;
    std::vector<float>    h_depression_bias;
};
 
struct HostPipelineBuffers {
    std::vector<float>    h_delayed_charge;
    std::vector<uint32_t> h_dirty_delayed_charge;
    std::vector<uint8_t>  h_spiked;
    std::vector<uint32_t> h_spiked_only;
    uint32_t              h_spiked_only_offset;
    std::vector<float>    h_trace_lut;
    std::vector<float> h_input;
    std::vector<uint32_t> h_output;
};

struct HostOrganismBuffers {
    HostGlobalParams       h_global_params_{};
    HostNeuronBuffers      h_neuron_buffers_{};
    HostCSRCSCBuffers      h_csr_csc_buffers_{};
    HostPipelineBuffers    h_pipeline_buffers_{};
};