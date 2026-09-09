#pragma once

#include "DNAviews.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <random>
#include <string>
#include <vector>

class Reproduction {
public:
    static void reproduce(int specie_id, const std::vector<OrganismLog>& organisms);
    static void mutateOrganism(const std::filesystem::path& organism_path);

private:
    static int allocateInnovationId(
        int source_node_innovation_id, int target_node_innovation_id, int generation);
    static int allocateInnovationId(int split_connection_innovation_id, int generation);
    static nlohmann::json newConnection(
        const nlohmann::json& dna_json,
        int source_node_innovation_id,
        int target_node_innovation_id,
        int generation,
        std::mt19937& gen);
    static float mutate(
        nlohmann::json& genome,
        const std::string& variable_name,
        float mutation_chance,
        float random_change_chance,
        const std::string& variable_type,
        std::mt19937& gen);
};
