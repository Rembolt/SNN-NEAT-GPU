#pragma once

#include "Constants.h"
#include "DNAviews.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>





struct SimpleGene {
    int32_t innovation_id;
    std::vector<std::pair<std::string, float>> params;
};



class Speciation {
public:

    /// behaviour distance in [0,1]; 1 when either side has no comparable trace
    static float compareOrganisms(
        const OutputLog& organism1,
        const OutputLog& organism2);
    /// reps to score on this generation's clip, so their traces stay comparable
    static std::vector<OrganismLog> repOrganisms();
    static int getSpecie(
        nlohmann::json& organism,
        const std::filesystem::path& resolved_path,
        nlohmann::json& world,
        const OutputLog& output_log,
        std::vector<OrganismLog>& rep_logs);
    static Specie getSpecie(int specie_id);
    static void updateSpecie(
        const nlohmann::json& organism,
        const std::filesystem::path& resolved_path,
        nlohmann::json& world);
    static int newSpecie(
        nlohmann::json& organism,
        const std::filesystem::path& resolved_path,
        nlohmann::json& world);
    static std::vector<OrganismLog> updateAdjustedFitness(
        std::vector<OrganismLog> organisms,
        std::vector<OrganismLog>& rep_logs);
    static void progressTest(const std::vector<OrganismLog>& progress_logs);
    static void updateWorldManager();
    static void saveWorldManager();
    static void updateOffspringShare();
    static void pruneDeadSpecies();
    static void deleteOrganism(const std::filesystem::path& dest);
    static void copyOrganism(std::string_view source_path_string, const std::filesystem::path& dest);
    static std::string copyOrganism(
        const std::filesystem::path& source_resolved_path,
        int specie_id,
        std::string_view suffix);
    static int allocateOrganismId();
    static void commitOrganismIds();
    static void setSizeThisGen(int specie_id, int size);

private:

    static bool isSeedDnaPath(const std::filesystem::path& resolved_path);
    // NOT USED: genome-structure comparison, replaced by behaviour traces
    static std::vector<SimpleGene> buildComparisonView(const nlohmann::json& organism);
    static float compareParams(
        const std::vector<std::pair<std::string, float>>& a,
        const std::vector<std::pair<std::string, float>>& b);
    static float compareGenomes(
        const nlohmann::json& organism1,
        const nlohmann::json& organism2,
        float c1,
        float c2,
        float c3,
        int min_genome_size);
};

