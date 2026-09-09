#include "DNAReader.h"
#include "Constants.h"
#include "HostFeedbackInput.h"
#include "HostMouseInput.h"
#include "util/Log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

DNAReader::DNAReader() = default;

void DNAReader::readDNAFile(const std::string& dna_file_path){

    //get dna.json
    std::filesystem::path path(dna_file_path);
    std::ifstream dna_file(path);

#ifdef SNN_GPU_PROJECT_ROOT
    if (!dna_file.is_open() && path.is_relative()) {
        path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / dna_file_path;
        dna_file.clear();
        dna_file.open(path);
    }
#endif

    if (!dna_file.is_open()) {
        Util::Errors::stop("Failed to open DNA file: " + path.string());
    }

    dna_json_ = nlohmann::json::parse(dna_file);
    h_organism_buffers_ = HostOrganismBuffers{};
    h_app_params_ = HostAppParams{};
    neuron_index_by_innovation_id_.clear();

    h_organism_buffers_.h_global_params_.h_output_size = dna_layout::k_output_slot_count;

    createNeurons();
    assignAppParams();
    assignGlobalParams();
    createCSRCSC();
    createTraceLUT();
    createPipelineBuffers();
}

void DNAReader::createNeurons(){
    auto& h_g = h_organism_buffers_.h_global_params_;
    auto& h_n = h_organism_buffers_.h_neuron_buffers_;
    const auto& neuron_genomes = dna_json_.at("neuron_genomes");

    h_g.h_num_output_slots = h_g.h_output_size;
    h_g.h_output_slot_neuron_counts =
        std::vector<uint32_t>(h_g.h_output_size, 0u);

    h_n.h_V.clear();
    h_n.h_W.clear();
    h_n.h_a.clear();
    h_n.h_b.clear();
    h_n.h_tau_w.clear();
    h_n.h_threshold_V.clear();
    h_n.h_v_rest.clear();
    h_n.h_neuron_kind.clear();
    h_n.h_neuron_kind_target.clear();
    h_n.h_last_delievery_update.clear();
    h_n.h_refract_end.clear();
    h_n.h_last_spike.clear();
    h_n.h_spike_rate.clear();
    h_n.h_spike_rate_target.clear();
    h_n.h_refractory_length.clear();

    for (const auto& neuron : neuron_genomes) {
        const uint32_t neuron_index = static_cast<uint32_t>(h_n.h_V.size());
        const float v_rest = neuron.at("v_rest").get<float>();
        const std::string neuron_type = neuron.at("neuron_type").get<std::string>();

        neuron_index_by_innovation_id_[neuron.at("innovation_id").get<uint32_t>()] = neuron_index;

        h_n.h_V.push_back(v_rest);
        h_n.h_W.push_back(0.0f);
        h_n.h_a.push_back(neuron.at("a_subthreshold_adaptation").get<float>());
        h_n.h_b.push_back(neuron.at("b_spike_triggered_adaptation").get<float>());
        h_n.h_tau_w.push_back(neuron.at("tau_w_adaptation_time_constant").get<float>());
        h_n.h_threshold_V.push_back(neuron.at("v_threshold").get<float>());
        h_n.h_v_rest.push_back(v_rest);
        h_n.h_last_delievery_update.push_back(0u);
        h_n.h_refract_end.push_back(0u);
        h_n.h_last_spike.push_back(std::numeric_limits<uint32_t>::max());
        h_n.h_spike_rate.push_back(0.0f);
        h_n.h_spike_rate_target.push_back(neuron.at("target_spike_rate").get<float>());
        h_n.h_refractory_length.push_back(neuron.at("refractory_length_ticks").get<uint32_t>());

        if (neuron_type == "hidden") {
            h_n.h_neuron_kind.push_back(dna_layout::k_neuron_kind_hidden);
            h_n.h_neuron_kind_target.push_back(neuron.value("neuron_type_target", dna_layout::k_no_kind_target));
        } else if (neuron_type == "input") {
            const uint32_t input_source = neuron.at("neuron_type_target").get<uint32_t>();
            h_n.h_neuron_kind.push_back(dna_layout::k_neuron_kind_input);
            h_n.h_neuron_kind_target.push_back(input_source);
        } else if (neuron_type == "output") {
            h_n.h_neuron_kind.push_back(dna_layout::k_neuron_kind_output);
            const uint32_t output_slot = neuron.at("neuron_type_target").get<uint32_t>();
            if (output_slot >= h_g.h_output_size) {
                Util::Errors::stop("Output neuron target slot is out of range: " + std::to_string(output_slot));
            }
            h_n.h_neuron_kind_target.push_back(output_slot);
            ++h_g.h_output_slot_neuron_counts[output_slot];
        } else if (neuron_type == "bias") {
            h_n.h_neuron_kind.push_back(dna_layout::k_neuron_kind_bias);
            h_n.h_neuron_kind_target.push_back(neuron.value("neuron_type_target", dna_layout::k_no_kind_target));
        } else {
            Util::Errors::stop("Unknown neuron_type: " + neuron_type);
        }
    }
}

void DNAReader::assignAppParams(){
    auto& h_g = h_organism_buffers_.h_global_params_;
    auto& h_p = h_organism_buffers_.h_pipeline_buffers_;
    auto& h_app = h_app_params_;
    const auto& global_genome = dna_json_.at("global_genome");

    h_app.h_screen_input_size = 0;
    h_app.h_mouse_input_size =
        static_cast<int32_t>(Util::hostInputStructFloatSlotCount<HostMouseInput>());
    h_app.h_error_input_size =
        static_cast<int32_t>(Util::hostInputStructFloatSlotCount<HostFeedbackInput>());

    h_app.h_mouse_output_axis_decay = global_genome.at("output_axis_decay").get<float>();
    h_app.x_Bins = static_cast<int32_t>(dna_layout::k_mouse_output_x_bins);
    h_app.y_Bins = static_cast<int32_t>(dna_layout::k_mouse_output_y_bins);

    h_app.h_feedback_temporal_decay_ticks =
        global_genome.value("feedback_temporal_decay_ticks", 120u);

    h_g.h_num_of_input_neurons = static_cast<uint32_t>(
        h_app.h_screen_input_size + h_app.h_mouse_input_size + h_app.h_error_input_size);

    h_p.h_input = std::vector<float>(
        static_cast<std::size_t>(h_app.h_screen_input_size)
        + static_cast<std::size_t>(h_app.h_mouse_input_size)
        + static_cast<std::size_t>(h_app.h_error_input_size),
        0.0f);


    
}

void DNAReader::assignGlobalParams(){

    auto& h_g = h_organism_buffers_.h_global_params_;
    const auto& global_genome = dna_json_.at("global_genome");

    h_g.h_current_tick = 0u;
    h_g.h_dt_ms = 0.1f;
    h_g.h_num_output_slots = h_g.h_output_size;
    if (h_g.h_output_slot_neuron_counts.empty()) {
        h_g.h_output_slot_neuron_counts =
            std::vector<uint32_t>(h_g.h_output_size, 0u);
    }
    h_g.h_num_neurons = static_cast<uint32_t>(h_organism_buffers_.h_neuron_buffers_.h_V.size());
    h_g.h_words_per_row = (h_g.h_num_neurons + 31u) / 32u;
    h_g.h_trace_tau = global_genome.at("max_trace_tick").get<float>();
    h_g.h_max_trace_len = global_genome.at("max_trace_tick").get<uint32_t>();
    h_g.h_max_delay = global_genome.at("max_delay").get<uint32_t>();
    h_g.h_click_distance_weight = global_genome.at("click_distance_weight").get<float>();
    h_g.h_homeo_weight_mod = global_genome.at("homeostasis_weight_mod").get<float>();
    h_g.h_homeo_pot_mod = global_genome.at("homeostasis_pot_bias_mod").get<float>();
    h_g.h_homeo_dep_mod = global_genome.at("homeostasis_dep_bias_mod").get<float>();
    h_g.h_homeo_thresh_mod = global_genome.at("homeostasis_threshold_mod").get<float>();
    h_g.h_homeo_rate_max_distance = global_genome.at("homeostasis_rate_max_distance").get<float>();
    h_g.h_ticks_between_homeostasis = global_genome.value("ticks_between_homeostasis", 1000u);
    h_g.h_capacitance_C = global_genome.at("membrane_capacitance").get<float>();
    h_g.h_leak_g = global_genome.at("leak_conductance").get<float>();
    h_g.h_leak_E = global_genome.at("leak_reversal_potential").get<float>();
    h_g.h_slope_delta_T = global_genome.at("slope_factor").get<float>();
    h_g.h_v_peak = global_genome.at("spike_peak_voltage").get<float>();
    h_g.h_inhibitory_spike_rate = 0u;
    h_g.h_excitatory_spike_rate = 0u;
    
}

void DNAReader::createCSRCSC(){
    auto& h_c = h_organism_buffers_.h_csr_csc_buffers_;
    auto& h_g = h_organism_buffers_.h_global_params_;
    const uint32_t num_neurons = h_g.h_num_neurons;

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> forward_adjecent_lists(num_neurons);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> backward_adjecent_lists(num_neurons);
    std::vector<const nlohmann::json*> connections_by_index;

    for (const auto& connection : dna_json_.at("connection_genomes")) {
        if (connection.contains("enabled") && !connection.at("enabled").get<bool>()) {
            continue;
        }

        const uint32_t connection_index = static_cast<uint32_t>(connections_by_index.size());
        const uint32_t source = neuron_index_by_innovation_id_.at(connection.at("source_node_innovation_id").get<uint32_t>());
        const uint32_t target = neuron_index_by_innovation_id_.at(connection.at("target_node_innovation_id").get<uint32_t>());

        connections_by_index.push_back(&connection);
        forward_adjecent_lists[source].push_back({target, connection_index});
    }

    h_c.h_csr_row_ptr.clear();
    h_c.h_csr_col_idx.clear();
    h_c.h_weight.clear();
    h_c.h_delay.clear();
    h_c.h_potentiation_bias.clear();
    h_c.h_depression_bias.clear();

    h_c.h_csr_row_ptr.push_back(0u);
    for (uint32_t source = 0; source < num_neurons; ++source) {
        for (const auto& [target, connection_index] : forward_adjecent_lists[source]) {
            const auto& connection = *connections_by_index[connection_index];
            const uint32_t csr_index = static_cast<uint32_t>(h_c.h_csr_col_idx.size());

            h_c.h_csr_col_idx.push_back(target);
            h_c.h_weight.push_back(connection.at("initial_weight").get<float>());
            //delay_ticks cannot exceed the organism's max_delay circular buffer depth
            h_c.h_delay.push_back(std::min(
                connection.at("delay_ticks").get<float>(),
                static_cast<float>(h_g.h_max_delay)));
            h_c.h_potentiation_bias.push_back(connection.at("potentiation_bias").get<float>());
            h_c.h_depression_bias.push_back(connection.at("depression_bias").get<float>());
            backward_adjecent_lists[target].push_back({source, csr_index});
        }
        h_c.h_csr_row_ptr.push_back(static_cast<uint32_t>(h_c.h_csr_col_idx.size()));
    }

    h_c.h_csc_row_ptr.clear();
    h_c.h_csc_col_idx.clear();
    h_c.h_csc_to_csr_index_mapping.clear();

    h_c.h_csc_row_ptr.push_back(0u);
    for (uint32_t target = 0; target < num_neurons; ++target) {
        for (const auto& [source, csr_index] : backward_adjecent_lists[target]) {
            h_c.h_csc_col_idx.push_back(source);
            h_c.h_csc_to_csr_index_mapping.push_back(csr_index);
        }
        h_c.h_csc_row_ptr.push_back(static_cast<uint32_t>(h_c.h_csc_col_idx.size()));
    }

    h_g.h_num_synapses = static_cast<uint32_t>(h_c.h_weight.size());
}

void DNAReader::createTraceLUT(){

    auto& h_g = h_organism_buffers_.h_global_params_;
    auto& h_p = h_organism_buffers_.h_pipeline_buffers_;

    h_p.h_trace_lut = std::vector<float>(h_g.h_max_trace_len, 0.0f);
    std::vector<float> h_trace_lut(h_g.h_max_trace_len);
    const float tau = static_cast<float>(h_g.h_trace_tau);
    for (uint32_t i = 0; i < h_g.h_max_trace_len; i++) {
        h_p.h_trace_lut[i] = (tau > 0.0f)
            ? std::exp(-static_cast<float>(i) / tau)
            : 1.0f;
    }
}

void DNAReader::createPipelineBuffers(){
    auto& h_g = h_organism_buffers_.h_global_params_;
    auto& h_p = h_organism_buffers_.h_pipeline_buffers_;

    h_p.h_dirty_delayed_charge = std::vector<uint32_t>(
        static_cast<std::size_t>(h_g.h_max_delay) * ((h_g.h_num_neurons + 31u) / 32u), 0u);
    h_p.h_spiked = std::vector<uint8_t>(h_g.h_num_neurons, 0u);
    h_p.h_spiked_only = std::vector<uint32_t>(h_g.h_num_neurons, 0u);
    h_p.h_spiked_only_offset = 0u;
    h_p.h_delayed_charge = std::vector<float>(
        static_cast<std::size_t>(h_g.h_max_delay) * h_g.h_num_neurons, 0.0f);

    h_p.h_output = std::vector<uint32_t>(h_g.h_num_output_slots, 0u);

    

}
