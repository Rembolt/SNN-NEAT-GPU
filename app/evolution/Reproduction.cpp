#include "Reproduction.h"
#include "Constants.h"
#include "Speciation.h"
#include "util/Log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <vector>


int Reproduction::allocateInnovationId(
    int source_node_innovation_id, int target_node_innovation_id, int generation)
{
    const std::filesystem::path innovation_path =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "app/evolution/innovation_ids.json";
    nlohmann::json innovation_ids_json = Util::JsonAtomicOpen(innovation_path);
    const std::string connection_key =
        std::to_string(source_node_innovation_id) + ","
        + std::to_string(target_node_innovation_id) + ","
        + std::to_string(generation);
    if (innovation_ids_json["connections"].contains(connection_key)) {
        return innovation_ids_json["connections"][connection_key];
    }
    const int innovation_id = innovation_ids_json["next_id"];
    innovation_ids_json["next_id"] = innovation_id + 1;
    innovation_ids_json["connections"][connection_key] = innovation_id;
    Util::JsonAtomicWrite(innovation_path, innovation_ids_json);
    return innovation_id;
}

int Reproduction::allocateInnovationId(int split_connection_innovation_id, int generation) {
    if(split_connection_innovation_id == -1) {
        Util::Errors::stop("Reproduction: split_connection_innovation_id is -1");
        return -1;
    }
    const std::filesystem::path innovation_path =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "app/evolution/innovation_ids.json";
    nlohmann::json innovation_ids_json = Util::JsonAtomicOpen(innovation_path);
    if (!innovation_ids_json["neurons"].is_object()) {
        innovation_ids_json["neurons"] = nlohmann::json::object();
    }
    // mutated neurons: "split_connection_id,generation" -> neuron innovation id
    const std::string neuron_key =
        std::to_string(split_connection_innovation_id) + "," + std::to_string(generation);
    if (innovation_ids_json["neurons"].contains(neuron_key)) {
        return innovation_ids_json["neurons"][neuron_key].get<int>();
    }
    const int innovation_id = innovation_ids_json["next_id"];
    innovation_ids_json["next_id"] = innovation_id + 1;
    innovation_ids_json["neurons"][neuron_key] = innovation_id;
    Util::JsonAtomicWrite(innovation_path, innovation_ids_json);
    return innovation_id;
}

nlohmann::json Reproduction::newConnection(
    const nlohmann::json& dna_json,
    int source_node_innovation_id,
    int target_node_innovation_id,
    int generation,
    std::mt19937& gen)
{
    if (target_node_innovation_id == -1) {
        return nlohmann::json::object();
    }
    // use preferred range for new param init wherever param_bounds is defined
    auto pref = [](const std::string& n) -> std::pair<float, float> {
        const auto it = organism_ranges::param_bounds.find(n);
        return it != organism_ranges::param_bounds.end()
            ? std::pair<float,float>{it->second.preferred_min, it->second.preferred_max}
            : organism_ranges::genome_ranges.at(n);
    };
    nlohmann::json new_connection = nlohmann::json::object();
    new_connection["innovation_id"] =
        allocateInnovationId(source_node_innovation_id, target_node_innovation_id, generation);
    new_connection["source_node_innovation_id"] = source_node_innovation_id;
    new_connection["target_node_innovation_id"] = target_node_innovation_id;
    new_connection["enabled"] = true;
    new_connection["initial_weight"] = std::uniform_real_distribution<float>(
        pref("initial_weight").first, pref("initial_weight").second)(gen);
    //delay_ticks cannot exceed the organism's max_delay circular buffer depth
    const int organism_max_delay = dna_json["global_genome"].at("max_delay").get<int>();
    const int delay_min = static_cast<int>(pref("delay_ticks").first);
    const int delay_max = std::max(delay_min, std::min(static_cast<int>(pref("delay_ticks").second), organism_max_delay));
    new_connection["delay_ticks"] = std::uniform_int_distribution<int>(delay_min, delay_max)(gen);
    new_connection["potentiation_bias"] = std::uniform_real_distribution<float>(
        pref("potentiation_bias").first, pref("potentiation_bias").second)(gen);
    new_connection["depression_bias"] = std::uniform_real_distribution<float>(
        pref("depression_bias").first, pref("depression_bias").second)(gen);
    return new_connection;
}

void Reproduction::reproduce(int specie_id, const std::vector<OrganismLog>& organisms)
{
    if (organisms.empty()) {
        return;
    }

    const Specie specie = Speciation::getSpecie(specie_id);
    const int offspring_count = specie.offspring_count;
    if (offspring_count <= 0) {
        return;
    }

    std::vector<OrganismLog> sorted = organisms;
    std::sort(sorted.begin(), sorted.end(), [](const OrganismLog& a, const OrganismLog& b) {
        return a.adjusted_fitness > b.adjusted_fitness;
    });

    const std::filesystem::path output_dir =
        std::filesystem::path(sorted.front().organism_path).parent_path();

    int slot = 0;
    const int elite_count = offspring_count > 17 ? 2 : offspring_count > 1 ? 1 : 0;
    for (; slot < elite_count; ++slot) {
        const std::filesystem::path elite_dest =
            output_dir / (std::to_string(Speciation::allocateOrganismId()) + ".json");
        //pick top organisms by fitness, capped to available
        Speciation::copyOrganism(sorted[std::min(slot, (int)sorted.size() - 1)].organism_path, elite_dest);
        nlohmann::json elite_json = Util::JsonAtomicOpen(elite_dest);
        auto& elite_genome = elite_json["global_genome"];
        elite_genome["generation"] = elite_genome.value("generation", 0) + 1;
        Util::JsonAtomicWrite(elite_dest, elite_json, 4);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    for (; slot < offspring_count; ++slot) {
        std::uniform_int_distribution<> tour_size_dist(3, 4);
        const int tour_size = std::min(tour_size_dist(gen), static_cast<int>(sorted.size()));
        std::vector<OrganismLog> potential_parents;
        //sample preserves relative order from sorted source, so front() is already best
        std::sample(
            sorted.begin(),
            sorted.end(),
            std::back_inserter(potential_parents),
            static_cast<std::size_t>(tour_size),
            gen);

        const std::filesystem::path child_dest =
            output_dir / (std::to_string(Speciation::allocateOrganismId()) + ".json");
        Speciation::copyOrganism(potential_parents.front().organism_path, child_dest);
        mutateOrganism(child_dest);
    }
    Speciation::commitOrganismIds();
}

float Reproduction::mutate(
    nlohmann::json& genome,
    const std::string& variable_name,
    float mutation_chance,
    float random_change_chance,
    const std::string& variable_type,
    std::mt19937& gen)
{
    float current_value = genome.at(variable_name).get<float>();

    // resolve bounds: param_bounds supplies preferred (for sigma/random) and absolute (for clamp)
    const auto pb_it = organism_ranges::param_bounds.find(variable_name);
    const bool has_pb = pb_it != organism_ranges::param_bounds.end();
    const float abs_min = has_pb ? pb_it->second.absolute_min : organism_ranges::genome_ranges.at(variable_name).first;
    const float abs_max = has_pb ? pb_it->second.absolute_max : organism_ranges::genome_ranges.at(variable_name).second;
    const float pref_min = has_pb ? pb_it->second.preferred_min : abs_min;
    const float pref_max = has_pb ? pb_it->second.preferred_max : abs_max;

    std::uniform_real_distribution<float> unit_dist(0.f, 1.f);

    if (unit_dist(gen) > mutation_chance) {
        current_value = std::clamp(current_value, abs_min, abs_max);
        return current_value;
    }

    if (unit_dist(gen) < random_change_chance) {
        if (variable_type == "integer") {
            std::uniform_int_distribution<int> int_dist(static_cast<int>(pref_min), static_cast<int>(pref_max));
            current_value = static_cast<float>(int_dist(gen));
        } else {
            std::uniform_real_distribution<float> float_dist(pref_min, pref_max);
            current_value = float_dist(gen);
        }
        current_value = std::clamp(current_value, abs_min, abs_max);
        return current_value;
    }

    if (variable_type == "integer" || variable_type == "decimal") {
        static std::normal_distribution<float> standard_normal(0.f, 1.f);
        float sigma;
        if (variable_type == "integer") {
            std::uniform_real_distribution<float> mod_dist(
                static_cast<float>(mutation::k_integer_mutation_mod_min),
                static_cast<float>(mutation::k_integer_mutation_mod_max));
            sigma = mod_dist(gen);
        } else if (has_pb && pb_it->second.mutation_scale_pct > 0.0f) {
            // per-param biological scale: sigma = pct * preferred_range
            sigma = pb_it->second.mutation_scale_pct * (pref_max - pref_min);
        } else {
            std::uniform_real_distribution<float> mod_dist(
                mutation::k_decimal_mutation_mod_min, mutation::k_decimal_mutation_mod_max);
            sigma = mod_dist(gen) * (abs_max - abs_min);
        }
        current_value = current_value + standard_normal(gen) * sigma;
        if (variable_type == "integer") {
            current_value = std::round(current_value);
        }
        current_value = std::clamp(current_value, abs_min, abs_max);
        return current_value;
    }
    current_value = std::clamp(current_value, abs_min, abs_max);
    return current_value;
}

void Reproduction::mutateOrganism(const std::filesystem::path& organism_path)
{
    nlohmann::json dna_json = Util::JsonAtomicOpen(organism_path);

    auto& global_genome = dna_json["global_genome"];
    const int next_generation = global_genome.value("generation", 0) + 1;
    global_genome["species_id"] = -1;
    global_genome["organism_id"] = std::stoi(organism_path.stem().string());
    global_genome["generation"] = next_generation;
    global_genome["fitness"] = 0.0;
    global_genome["adjusted_fitness"] = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> unit_dist(0.f, 1.f);

    float mutation_chance = mutation::k_global_genome_mutation_chance;
    float random_change_chance = mutation::k_global_genome_random_change_chance;

    //global mutation
    global_genome["membrane_capacitance"] = mutate(global_genome, "membrane_capacitance", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["leak_conductance"] = mutate(global_genome, "leak_conductance", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["leak_reversal_potential"] = mutate(global_genome, "leak_reversal_potential", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["slope_factor"] = mutate(global_genome, "slope_factor", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["spike_peak_voltage"] = mutate(global_genome, "spike_peak_voltage", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["homeostasis_weight_mod"] = mutate(global_genome, "homeostasis_weight_mod", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["homeostasis_pot_bias_mod"] = mutate(global_genome, "homeostasis_pot_bias_mod", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["homeostasis_dep_bias_mod"] = mutate(global_genome, "homeostasis_dep_bias_mod", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["homeostasis_threshold_mod"] = mutate(global_genome, "homeostasis_threshold_mod", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["homeostasis_rate_max_distance"] = mutate(global_genome, "homeostasis_rate_max_distance", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["ticks_between_homeostasis"] = static_cast<int>(mutate(global_genome, "ticks_between_homeostasis", mutation_chance, random_change_chance, "integer", gen));
    global_genome["max_trace_tick"] = static_cast<int>(mutate(global_genome, "max_trace_tick", mutation_chance, random_change_chance, "integer", gen));
    global_genome["max_delay"] = static_cast<int>(mutate(global_genome, "max_delay", mutation_chance, random_change_chance, "integer", gen));
    global_genome["click_distance_weight"] = mutate(global_genome, "click_distance_weight", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["output_axis_decay"] = mutate(global_genome, "output_axis_decay", mutation_chance, random_change_chance, "decimal", gen);
    global_genome["feedback_temporal_decay_ticks"] = static_cast<int>(mutate(global_genome, "feedback_temporal_decay_ticks", mutation_chance, random_change_chance, "integer", gen));

    for (auto& neuron : dna_json["neuron_genomes"]) {
        if (neuron["neuron_type"] == "input") {
            mutation_chance = mutation::k_input_neuron_mutation_chance;
            random_change_chance = mutation::k_input_neuron_random_change_chance;
        } else if (neuron["neuron_type"] == "hidden") {
            mutation_chance = mutation::k_hidden_neuron_mutation_chance;
            random_change_chance = mutation::k_hidden_neuron_random_change_chance;
        } else if (neuron["neuron_type"] == "output") {
            mutation_chance = mutation::k_output_neuron_mutation_chance;
            random_change_chance = mutation::k_output_neuron_random_change_chance;
        } else if (neuron["neuron_type"] == "bias") {
            mutation_chance = mutation::k_bias_neuron_mutation_chance;
            random_change_chance = mutation::k_bias_neuron_random_change_chance;
        } else {
            mutation_chance = mutation::k_hidden_neuron_mutation_chance;
            random_change_chance = mutation::k_hidden_neuron_random_change_chance;
        }
        if(mutation_chance <=0.0f) {
            continue;
        }
        neuron["a_subthreshold_adaptation"] = mutate(neuron, "a_subthreshold_adaptation", mutation_chance, random_change_chance, "decimal", gen);
        neuron["b_spike_triggered_adaptation"] = mutate(neuron, "b_spike_triggered_adaptation", mutation_chance, random_change_chance, "decimal", gen);
        neuron["tau_w_adaptation_time_constant"] = mutate(neuron, "tau_w_adaptation_time_constant", mutation_chance, random_change_chance, "decimal", gen);
        neuron["v_threshold"] = mutate(neuron, "v_threshold", mutation_chance, random_change_chance, "decimal", gen);
        neuron["v_rest"] = mutate(neuron, "v_rest", mutation_chance, random_change_chance, "decimal", gen);
        neuron["refractory_length_ticks"] = static_cast<int>(mutate(neuron, "refractory_length_ticks", mutation_chance, random_change_chance, "integer", gen));
        neuron["target_spike_rate"] = mutate(neuron, "target_spike_rate", mutation_chance, random_change_chance, "decimal", gen);
    }
    mutation_chance = mutation::k_connection_genome_mutation_chance;
    random_change_chance = mutation::k_connection_genome_random_change_chance;

    //grab the freshly mutated max_delay so delay_ticks can never exceed the circular buffer depth
    const int organism_max_delay = global_genome.at("max_delay").get<int>();
    for (auto& connection : dna_json["connection_genomes"]) {
        connection["initial_weight"] = mutate(connection, "initial_weight", mutation_chance, random_change_chance, "decimal", gen);
        connection["delay_ticks"] = std::min(
            static_cast<int>(mutate(connection, "delay_ticks", mutation_chance, random_change_chance, "integer", gen)),
            organism_max_delay);
        connection["potentiation_bias"] = mutate(connection, "potentiation_bias", mutation_chance, random_change_chance, "decimal", gen);
        connection["depression_bias"] = mutate(connection, "depression_bias", mutation_chance, random_change_chance, "decimal", gen);
    }

    const auto& neuron_genomes = dna_json["neuron_genomes"];
    if (!neuron_genomes.empty() && unit_dist(gen) < mutation::k_new_connection_genome_chance) {
        std::uniform_int_distribution<int> neuron_index_dist(0, static_cast<int>(neuron_genomes.size()) - 1);

        //outputs cannot be a connection source, re-roll until a non-output is picked
        int source_node_innovation_id = -1;
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto& candidate = neuron_genomes[neuron_index_dist(gen)];
            if (candidate["neuron_type"] != "output") {
                source_node_innovation_id = candidate["innovation_id"];
                break;
            }
        }

        //inputs/bias cannot be a connection target, re-roll until a non-input/bias is picked
        int target_node_innovation_id = -1;
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto& candidate = neuron_genomes[neuron_index_dist(gen)];
            const std::string& ntype = candidate["neuron_type"];
            if (ntype != "input" && ntype != "bias") {
                target_node_innovation_id = candidate["innovation_id"];
                break;
            }
        }

        //same source/target pair in this generation maps to the same innovation id
        //skip if the organism already has that topology edge
        bool connection_exists = false;
        for (const auto& connection : dna_json["connection_genomes"]) {
            if (connection["source_node_innovation_id"] == source_node_innovation_id &&
                connection["target_node_innovation_id"] == target_node_innovation_id) {
                connection_exists = true;
                break;
            }
        }

        if (!connection_exists
            && source_node_innovation_id != -1
            && target_node_innovation_id != -1) {
            const nlohmann::json mutated_connection =
                newConnection(
                    dna_json, source_node_innovation_id, target_node_innovation_id, next_generation, gen);
            if (!mutated_connection.empty()) {
                dna_json["connection_genomes"].push_back(mutated_connection);
            }
        }
    }

    auto& connection_genomes = dna_json["connection_genomes"];
    if (!connection_genomes.empty() && unit_dist(gen) < mutation::k_disable_connection_genome_chance) {
        std::uniform_int_distribution<int> connection_index_dist(0, static_cast<int>(connection_genomes.size()) - 1);
        connection_genomes[connection_index_dist(gen)]["enabled"] = false;
    }

    if (!connection_genomes.empty() && unit_dist(gen) < mutation::k_enable_connection_genome_chance) {
        std::vector<int> disabled_indices;
        disabled_indices.reserve(connection_genomes.size());
        for (int i = 0; i < static_cast<int>(connection_genomes.size()); ++i) {
            if (!connection_genomes[i].value("enabled", true)) {
                disabled_indices.push_back(i);
            }
        }
        if (!disabled_indices.empty()) {
            std::uniform_int_distribution<int> disabled_index_dist(
                0, static_cast<int>(disabled_indices.size()) - 1);
            connection_genomes[disabled_indices[disabled_index_dist(gen)]]["enabled"] = true;
        }
    }

    if (!connection_genomes.empty() && unit_dist(gen) < mutation::k_new_neuron_genome_chance) {
        std::uniform_int_distribution<int> connection_index_dist(0, static_cast<int>(connection_genomes.size()) - 1);
        auto& split_connection = connection_genomes[connection_index_dist(gen)];
        const int source_node_innovation_id = split_connection["source_node_innovation_id"];
        const int target_node_innovation_id = split_connection["target_node_innovation_id"];

        nlohmann::json new_neuron = nlohmann::json::object();
        new_neuron["innovation_id"] =
            allocateInnovationId(split_connection["innovation_id"].get<int>(), next_generation);
        new_neuron["split_connection_id"] = split_connection["innovation_id"];
        //const int neuron_type_roll = std::uniform_int_distribution<int>(0, 99)(gen);
        //if (neuron_type_roll < 80) {
            new_neuron["neuron_type"] = "hidden";
            new_neuron["neuron_type_target"] = 0;
        //} else if (neuron_type_roll < 90) {
        //    new_neuron["neuron_type"] = "output";
        //    new_neuron["neuron_type_target"] = std::uniform_int_distribution<int>(
        //        0, static_cast<int>(dna_layout::k_output_slot_count) - 1)(gen);
        //} else if (neuron_type_roll < 97) {
        //    new_neuron["neuron_type"] = "input";
        //    new_neuron["neuron_type_target"] = std::uniform_int_distribution<int>(
        //        0, dna_layout::k_input_slot_count - 1)(gen);
        //} else {
        //    new_neuron["neuron_type"] = "bias";
        //    new_neuron["neuron_type_target"] = 0;
        //}
        // use preferred range for new param init wherever param_bounds is defined
        auto pref = [](const std::string& n) -> std::pair<float, float> {
            const auto it = organism_ranges::param_bounds.find(n);
            return it != organism_ranges::param_bounds.end()
                ? std::pair<float,float>{it->second.preferred_min, it->second.preferred_max}
                : organism_ranges::genome_ranges.at(n);
        };
        new_neuron["a_subthreshold_adaptation"] = std::uniform_real_distribution<float>(
            pref("a_subthreshold_adaptation").first, pref("a_subthreshold_adaptation").second)(gen);
        new_neuron["b_spike_triggered_adaptation"] = std::uniform_real_distribution<float>(
            pref("b_spike_triggered_adaptation").first, pref("b_spike_triggered_adaptation").second)(gen);
        new_neuron["tau_w_adaptation_time_constant"] = std::uniform_real_distribution<float>(
            pref("tau_w_adaptation_time_constant").first, pref("tau_w_adaptation_time_constant").second)(gen);
        new_neuron["v_threshold"] = std::uniform_real_distribution<float>(
            pref("v_threshold").first, pref("v_threshold").second)(gen);
        new_neuron["v_rest"] = std::uniform_real_distribution<float>(
            pref("v_rest").first, pref("v_rest").second)(gen);
        new_neuron["refractory_length_ticks"] = std::uniform_int_distribution<int>(
            static_cast<int>(pref("refractory_length_ticks").first),
            static_cast<int>(pref("refractory_length_ticks").second))(gen);
        new_neuron["target_spike_rate"] = std::uniform_real_distribution<float>(
            pref("target_spike_rate").first, pref("target_spike_rate").second)(gen);
        dna_json["neuron_genomes"].push_back(new_neuron);

        //NEAT split: disable original connection, bridge source -> new_neuron -> target
        //disable before push_back, growing connection_genomes invalidates the reference
        split_connection["enabled"] = false;

        const nlohmann::json split_in_connection =
            newConnection(
                dna_json,
                source_node_innovation_id,
                new_neuron["innovation_id"].get<int>(),
                next_generation,
                gen);
        if (!split_in_connection.empty()) {
            dna_json["connection_genomes"].push_back(split_in_connection);
        }
        const nlohmann::json split_out_connection =
            newConnection(
                dna_json,
                new_neuron["innovation_id"].get<int>(),
                target_node_innovation_id,
                next_generation,
                gen);
        if (!split_out_connection.empty()) {
            dna_json["connection_genomes"].push_back(split_out_connection);
        }
    }

    Util::JsonAtomicWrite(organism_path, dna_json, 4);
}
