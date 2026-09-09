#include "Speciation.h"
#include "Constants.h"
#include "util/Log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>



bool Speciation::isSeedDnaPath(const std::filesystem::path& resolved_path)
{
    std::error_code ec;
    const std::filesystem::path rel = std::filesystem::relative(
        resolved_path,
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "DNAs",
        ec);
    if (ec || rel.empty()) {
        return false;
    }
    for (const auto& part : rel) {
        if (part == "..") {
            return false;
        }
    }
    return true;
}

void Speciation::deleteOrganism(const std::filesystem::path& dest)
{
    //delete organism if it exists
    if (std::filesystem::exists(dest)) {
        std::error_code ec;
        std::filesystem::remove(dest, ec);
        if (ec) {
            Util::Errors::stop(
                std::string("Speciation: failed to delete species rep file: ")
                + dest.string()
                + ": "
                + ec.message());
        }
    }
}

void Speciation::copyOrganism(std::string_view source_path_string, const std::filesystem::path& dest)
{
    std::filesystem::path resolved_source(source_path_string);
    if (!std::filesystem::exists(resolved_source) && resolved_source.is_relative()) {
        resolved_source = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / resolved_source;
    }
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    const std::filesystem::path temp_path = dest.string() + ".tmp";
    std::filesystem::copy_file(resolved_source, temp_path, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        Util::Errors::stop(
            std::string("Speciation: failed to copy organism from ")
            + resolved_source.string()
            + " to "
            + temp_path.string()
            + ": "
            + ec.message());
    }
    Util::atomicFileReplace(temp_path, dest);
}

std::string Speciation::copyOrganism(
    const std::filesystem::path& source_resolved_path,
    int specie_id,
    std::string_view suffix)
{
    const std::filesystem::path species_dir =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species";
    if (!std::filesystem::exists(species_dir)) {
        Util::Errors::stop(
            std::string("Speciation: failed to find species folder: ")
            + species_dir.string());
    }
    const std::string filename = std::to_string(specie_id) + "-" + std::string(suffix) + ".json";
    const std::filesystem::path dest = species_dir / filename;
    copyOrganism(source_resolved_path.generic_string(), dest);
    return (std::filesystem::path("species") / filename).generic_string();
}

namespace {
nlohmann::json g_organism_id_world;
std::filesystem::path g_organism_id_world_path;
bool g_organism_id_world_loaded = false;
}  // namespace



int Speciation::allocateOrganismId()
{
    if (!g_organism_id_world_loaded) {
        g_organism_id_world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
        g_organism_id_world = Util::JsonAtomicOpen(g_organism_id_world_path, &g_organism_id_world_path);
        g_organism_id_world_loaded = true;
    }
    const int next_id = g_organism_id_world.at("next_organism_id").get<int>();
    g_organism_id_world["next_organism_id"] = next_id + 1;
    return next_id;
}

void Speciation::commitOrganismIds()
{
    if (!g_organism_id_world_loaded) {
        return;
    }
    Util::JsonAtomicWrite(g_organism_id_world_path, g_organism_id_world);
    g_organism_id_world_loaded = false;
}

std::vector<OrganismLog> Speciation::repOrganisms()
{
    std::filesystem::path world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    const nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    std::vector<OrganismLog> reps;
    reps.reserve(world.at("species").size());
    for (const auto& specie : world.at("species")) {
        OrganismLog log{};
        log.organism_path = (std::filesystem::path(SNN_GPU_PROJECT_ROOT)
            / specie.at("rep_path").get<std::string>()).generic_string();
        log.organism_id = specie.value("rep_id", 0);
        log.specie_id = specie.at("id").get<int>();
        reps.push_back(std::move(log));
    }
    return reps;
}

float Speciation::compareOrganisms(
    const OutputLog& organism1,
    const OutputLog& organism2)
{
    const std::size_t sample_count =
        std::min(organism1.samples.size(), organism2.samples.size());
    if (sample_count == 0u) {
        return 1.0f;
    }

    double position_sum = 0.0;
    double click_sum = 0.0;
    for (std::size_t i = 0; i < sample_count; ++i) {
        const OutputLogSample& sample_a = organism1.samples[i];
        const OutputLogSample& sample_b = organism2.samples[i];
        const double dx = static_cast<double>(sample_a.predicted_x - sample_b.predicted_x);
        const double dy = static_cast<double>(sample_a.predicted_y - sample_b.predicted_y);
        position_sum += std::sqrt(dx * dx + dy * dy);
        click_sum += 0.5
            * (std::abs(static_cast<double>(sample_a.left_click_rate - sample_b.left_click_rate))
               + std::abs(static_cast<double>(sample_a.right_click_rate - sample_b.right_click_rate)));
    }

    // both terms normalized to [0,1]: position by the unit-square diagonal, clicks by their own range
    const double inv_samples = 1.0 / static_cast<double>(sample_count);
    const float position_distance = static_cast<float>(
        position_sum * inv_samples / fitness_evaluation::k_jitter_distance_norm);
    const float click_distance = static_cast<float>(click_sum * inv_samples);
    return std::clamp(
        speciation::k_behavior_position_weight * position_distance
        + speciation::k_behavior_click_weight * click_distance,
        0.0f,
        1.0f);
}

// NOT USED: genome-structure comparison kept for reference
std::vector<SimpleGene> Speciation::buildComparisonView(const nlohmann::json& organism){
    const auto& global_genome = organism.at("global_genome");
    const auto& neuron_genomes = organism.at("neuron_genomes");
    const auto& connection_genomes = organism.at("connection_genomes");

    std::vector<SimpleGene> organism_view;
    organism_view.reserve(1 + neuron_genomes.size() + connection_genomes.size());
    organism_view.push_back(SimpleGene{
        static_cast<int32_t>(-1),
        {
        {"membrane_capacitance", global_genome.at("membrane_capacitance").get<float>()},
        {"leak_conductance", global_genome.at("leak_conductance").get<float>()},
        {"leak_reversal_potential", global_genome.at("leak_reversal_potential").get<float>()},
        {"slope_factor", global_genome.at("slope_factor").get<float>()},
        {"spike_peak_voltage", global_genome.at("spike_peak_voltage").get<float>()},
        {"homeostasis_weight_mod", global_genome.at("homeostasis_weight_mod").get<float>()},
        {"homeostasis_pot_bias_mod", global_genome.at("homeostasis_pot_bias_mod").get<float>()},
        {"homeostasis_dep_bias_mod", global_genome.at("homeostasis_dep_bias_mod").get<float>()},
        {"homeostasis_threshold_mod", global_genome.at("homeostasis_threshold_mod").get<float>()},
        {"homeostasis_rate_max_distance", global_genome.at("homeostasis_rate_max_distance").get<float>()},
        {"ticks_between_homeostasis", static_cast<float>(global_genome.value("ticks_between_homeostasis", 1000))},
        {"max_trace_tick", static_cast<float>(global_genome.at("max_trace_tick").get<int>())},
        {"max_delay", static_cast<float>(global_genome.at("max_delay").get<int>())},
        {"click_distance_weight", global_genome.at("click_distance_weight").get<float>()},
        {"output_axis_decay", global_genome.at("output_axis_decay").get<float>()},
        {"feedback_temporal_decay_ticks", static_cast<float>(global_genome.value("feedback_temporal_decay_ticks", 120))},
        }
    });

    for (const auto& neuron : neuron_genomes) {
        organism_view.push_back(SimpleGene{
            static_cast<int32_t>(neuron.at("innovation_id").get<uint32_t>()),
            {
                {"a_subthreshold_adaptation", neuron.at("a_subthreshold_adaptation").get<float>()},
                {"b_spike_triggered_adaptation", neuron.at("b_spike_triggered_adaptation").get<float>()},
                {"tau_w_adaptation_time_constant", neuron.at("tau_w_adaptation_time_constant").get<float>()},
                {"v_threshold", neuron.at("v_threshold").get<float>()},
                {"v_rest", neuron.at("v_rest").get<float>()},
                {"refractory_length_ticks", static_cast<float>(neuron.at("refractory_length_ticks").get<int>())},
                {"target_spike_rate", neuron.at("target_spike_rate").get<float>()}
            }});
    }
    for (const auto& connection : connection_genomes) {
        organism_view.push_back(SimpleGene{
            static_cast<int32_t>(connection.at("innovation_id").get<uint32_t>()),
            {
                {"initial_weight", connection.at("initial_weight").get<float>()},
                {"delay_ticks", static_cast<float>(connection.at("delay_ticks").get<int>())},
                {"potentiation_bias", connection.at("potentiation_bias").get<float>()},
                {"depression_bias", connection.at("depression_bias").get<float>()},
                {"enabled", connection.contains("enabled") && connection.at("enabled").get<bool>() ? 1.0f : 0.0f},
            }});
    }
    std::sort(
        organism_view.begin(),
        organism_view.end(),
        [](const SimpleGene& a, const SimpleGene& b){
            return a.innovation_id < b.innovation_id;
        }
    );
    return organism_view;
}

float Speciation::compareParams(
    const std::vector<std::pair<std::string, float>>& a,
    const std::vector<std::pair<std::string, float>>& b)
{
    if (a.empty()) {
        return 0.0f;
    }
    if (a.size() != b.size()) {
        return 1.0f;
    }

    float distance = 0.0f;
    for (const auto& param_a : a) {
        const auto param_b = std::find_if(
            b.begin(),
            b.end(),
            [&](const std::pair<std::string, float>& entry) {
                return entry.first == param_a.first;
            });
        if (param_b == b.end()) {
            return 1.0f;
        }
        // use preferred range as normalization so biologically meaningful differences register
        float range;
        const auto pb_it = organism_ranges::param_bounds.find(param_a.first);
        if (pb_it != organism_ranges::param_bounds.end()) {
            range = std::abs(pb_it->second.preferred_max - pb_it->second.preferred_min);
        } else {
            const auto& rp = organism_ranges::genome_ranges.at(param_a.first);
            range = std::abs(rp.first - rp.second);
        }
        distance += range > 0.0f
            ? std::abs(param_a.second - param_b->second) / range
            : 0.0f;
    }
    return distance / static_cast<float>(a.size());
}

// NOT USED: genome-structure comparison kept for reference
float Speciation::compareGenomes(
    const nlohmann::json& organism1,
    const nlohmann::json& organism2,
    float c1,
    float c2,
    float c3,
    int min_genome_size)
{
    const std::vector<SimpleGene> view_a = buildComparisonView(organism1);
    const std::vector<SimpleGene> view_b = buildComparisonView(organism2);

    const int32_t max_innov_a = view_a.back().innovation_id;
    const int32_t max_innov_b = view_b.back().innovation_id;

    int disjoint = 0;
    int excess = 0;
    float param_diff_sum = 0.0f;
    int matching_gene_count = 0;

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < view_a.size() && j < view_b.size()) {
        const SimpleGene& gene_a = view_a[i];
        const SimpleGene& gene_b = view_b[j];
        if (gene_a.innovation_id == gene_b.innovation_id) {
            param_diff_sum += compareParams(gene_a.params, gene_b.params);
            ++matching_gene_count;
            ++i;
            ++j;
            continue;
        }

        if (gene_a.innovation_id < gene_b.innovation_id) {
            if (gene_a.innovation_id <= max_innov_b) {
                ++disjoint;
            } else {
                ++excess;
            }
            ++i;
            continue;
        }

        if (gene_b.innovation_id <= max_innov_a) {
            ++disjoint;
        } else {
            ++excess;
        }
        ++j;
    }

    while (i < view_a.size()) {
        if (view_a[i].innovation_id > max_innov_b) {
            ++excess;
        } else {
            ++disjoint;
        }
        ++i;
    }
    while (j < view_b.size()) {
        if (view_b[j].innovation_id > max_innov_a) {
            ++excess;
        } else {
            ++disjoint;
        }
        ++j;
    }

    const int genome_size = std::max(
        static_cast<int>(view_a.size()) - 1,
        static_cast<int>(view_b.size()) - 1);
    const float normalization = genome_size >= min_genome_size
        ? static_cast<float>(genome_size)
        : 1.0f;

    const float avg_param_diff = matching_gene_count > 0
        ? param_diff_sum / static_cast<float>(matching_gene_count)
        : 0.0f;

    return (c1 * static_cast<float>(excess) + c2 * static_cast<float>(disjoint)) / normalization
        + c3 * avg_param_diff;
}

int Speciation::getSpecie(
    nlohmann::json& organism,
    const std::filesystem::path& resolved_path,
    nlohmann::json& world,
    const OutputLog& output_log,
    std::vector<OrganismLog>& rep_logs)
{

    const float threshold = world["speciation"].value("compatibility_threshold", speciation::k_compatibility_threshold);

    for (auto& specie : world.at("species")) {
        // rep trace comes from this generation's clip run, species without one cannot be joined
        const auto rep_log = std::find_if(
            rep_logs.begin(),
            rep_logs.end(),
            [&](const OrganismLog& entry) {
                return entry.specie_id == specie.at("id").get<int>();
            });
        if (rep_log == rep_logs.end()) {
            continue;
        }

        const float differentiation = compareOrganisms(output_log, rep_log->output_log);
        if (differentiation < threshold) {
            const int specie_id = specie.at("id").get<int>();
            // running mean of how far members drift from the rep behaviour this gen
            const int samples = specie.value("differentiation_samples", 0) + 1;
            const float mean = specie.value("specie_differentiation", 0.0f);
            specie["differentiation_samples"] = samples;
            specie["specie_differentiation"] = mean + (differentiation - mean) / static_cast<float>(samples);
            organism.at("global_genome")["species_id"] = specie_id;
            if (!isSeedDnaPath(resolved_path)) {
                Util::JsonAtomicWrite(resolved_path, organism);
            }
            return specie_id;
        }
    }

    const int specie_id = newSpecie(organism, resolved_path, world);
    // founder already ran this clip; later organisms can join this niche
    rep_logs.push_back(OrganismLog{
        resolved_path.generic_string(),
        organism.at("global_genome").at("organism_id").get<int>(),
        organism.at("global_genome").value("generation", 0),
        specie_id,
        organism.at("global_genome").value("fitness", 0.0f),
        0.0f,
        ScoreFactors{},
        output_log});
    return specie_id;
}

void Speciation::updateSpecie(
    const nlohmann::json& organism,
    const std::filesystem::path& resolved_path,
    nlohmann::json& world)
{
    const int specie_id = organism.at("global_genome").at("species_id").get<int>();
    const int organism_id = organism.at("global_genome").at("organism_id").get<int>();
    const float fitness = organism.at("global_genome").value("fitness", 0.0f);

    for (auto& specie : world.at("species")) {
        if (specie.at("id").get<int>() != specie_id) {
            continue;
        }
        specie["size_this_gen"] = specie.value("size_this_gen", 0) + 1;
        if (fitness > specie.value("best_fitness_this_gen", 0.0f)) {
            specie["best_fitness_this_gen"] = fitness;
            specie["rep_path"] = copyOrganism(resolved_path, specie_id, "rep");
            specie["rep_id"] = organism_id;
            specie["rep_fitness"] = fitness;
        }
        if (fitness > specie.value("best_fitness_ever", 0.0f)) {
            specie["best_fitness_ever"] = fitness;
        }
        break;
    }
}

int Speciation::newSpecie(
    nlohmann::json& organism,
    const std::filesystem::path& resolved_path,
    nlohmann::json& world)
{
    const int specie_id = world.at("next_species_id").get<int>();
    organism.at("global_genome")["species_id"] = specie_id;
    if (!isSeedDnaPath(resolved_path)) {
        Util::JsonAtomicWrite(resolved_path, organism);
    }

    const int generation = organism.at("global_genome").value("generation", 0);
    const int organism_id = organism.at("global_genome").at("organism_id").get<int>();
    const float fitness = organism.at("global_genome").value("fitness", 0.0f);
    const std::string rep_path = copyOrganism(resolved_path, specie_id, "rep");
    copyOrganism(resolved_path, specie_id, "legend");

    world.at("species").push_back({
        {"id", specie_id},
        {"rep_path", rep_path},
        {"rep_id", organism_id},
        {"rep_fitness", fitness},
        {"created_generation", generation},
        {"size_this_gen", 0},
        {"best_fitness_this_gen", fitness},
        {"best_fitness_ever", fitness},
        {"stagnation_generations", 0},
        {"score_on_progress_test", 0.0f},
        {"top_one_representation_accuracy", 1.0f},
        {"specie_differentiation", 0.0f},
        {"differentiation_samples", 0},
    });
    world["next_species_id"] = specie_id + 1;

    return specie_id;
}

void Speciation::progressTest(const std::vector<OrganismLog>& progress_logs)
{
    std::filesystem::path world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    for (auto& specie : world.at("species")) {
        const int specie_id = specie.at("id").get<int>();
        // best generalizing organism of the specie, plus its training-fitness leader
        const OrganismLog* champion = nullptr;
        const OrganismLog* top_one = nullptr;
        for (const OrganismLog& log : progress_logs) {
            if (log.specie_id != specie_id) {
                continue;
            }
            if (top_one == nullptr) {
                top_one = &log;
            }
            if (champion == nullptr || log.raw_fitness > champion->raw_fitness) {
                champion = &log;
            }
        }
        // 0.95 means you'd have recorded a progress score 5% below the best of the three.
        if (champion != nullptr && champion->raw_fitness > 0.0f) {
            specie["top_one_representation_accuracy"] = top_one->raw_fitness / champion->raw_fitness;
        }
        // untested species (no members this gen) age like a specie that failed to progress
        if (champion == nullptr
            || champion->raw_fitness <= specie.value("score_on_progress_test", 0.0f)) {
            specie["stagnation_generations"] = specie.value("stagnation_generations", 0) + 1;
            continue;
        }
        specie["score_on_progress_test"] = champion->raw_fitness;
        specie["stagnation_generations"] = 0;
        copyOrganism(std::filesystem::path(champion->organism_path), specie_id, "legend");
        Util::Logs::write(
            "Speciation: specie " + std::to_string(specie_id)
            + " progressed to " + std::to_string(champion->raw_fitness)
            + " on progress test\n");
    }

    Util::JsonAtomicWrite(world_path, world);
}

std::vector<OrganismLog> Speciation::updateAdjustedFitness(
    std::vector<OrganismLog> organisms,
    std::vector<OrganismLog>& rep_logs){
    //for every organism in this generation getSpecie(organism)
    std::filesystem::path world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    // parse once; assign against frozen reps, then elect champs after membership is fixed
    struct LoadedOrganism {
        std::filesystem::path path;
        nlohmann::json organism;
    };
    std::vector<LoadedOrganism> loaded;
    loaded.reserve(organisms.size());
    for (OrganismLog& log : organisms) {
        LoadedOrganism entry;
        entry.organism = Util::JsonAtomicOpen(std::filesystem::path(log.organism_path), &entry.path);
        entry.organism.at("global_genome")["fitness"] = log.raw_fitness;
        loaded.push_back(std::move(entry));
    }

    // pass 1: assign (new-species founders become niches immediately;
    // existing champs are not rewritten until every organism has a species_id)
    for (std::size_t i = 0; i < organisms.size(); ++i) {
        organisms[i].specie_id = getSpecie(
            loaded[i].organism,
            loaded[i].path,
            world,
            organisms[i].output_log,
            rep_logs);
    }

    // pass 2: sizes, gen champs, and legends
    for (std::size_t i = 0; i < organisms.size(); ++i) {
        // seed DNAs are not rewritten in getSpecie; restore assignment from the log
        loaded[i].organism.at("global_genome")["species_id"] = organisms[i].specie_id;
        updateSpecie(loaded[i].organism, loaded[i].path, world);
    }

    //calculate adjusted fitness = raw / species_size(this gen)
    std::unordered_map<int, ScoreFactors> score_factor_sums;
    std::unordered_map<int, int> score_factor_counts;
    for (OrganismLog& log : organisms) {
        const int specie_id = log.specie_id;
        const float fitness = log.raw_fitness;
        const ScoreFactors& sf = log.score_factors;
        ScoreFactors& sf_sum = score_factor_sums[specie_id];
        sf_sum.milliseconds_per_tick += sf.milliseconds_per_tick;
        sf_sum.mean_spike_density += sf.mean_spike_density;
        sf_sum.max_spike_density += sf.max_spike_density;
        sf_sum.inhibitory_excitatory_ratio += sf.inhibitory_excitatory_ratio;
        sf_sum.snn_neuron_synapse_ratio += sf.snn_neuron_synapse_ratio;
        sf_sum.mean_last_click_prediction_error += sf.mean_last_click_prediction_error;
        sf_sum.consistency_last_click_prediction_error += sf.consistency_last_click_prediction_error;
        sf_sum.mean_last_click_left_hits += sf.mean_last_click_left_hits;
        sf_sum.mean_last_click_right_hits += sf.mean_last_click_right_hits;
        sf_sum.mean_click_type_decisiveness += sf.mean_click_type_decisiveness;
        sf_sum.mean_certainty += sf.mean_certainty;
        sf_sum.calibration_error += sf.calibration_error;
        sf_sum.soft_jitter += sf.soft_jitter;
        sf_sum.medium_jitter += sf.medium_jitter;
        sf_sum.hard_jitter += sf.hard_jitter;
        sf_sum.learning += sf.learning;
        ++score_factor_counts[specie_id];
        for (auto& specie : world.at("species")) {
            if (specie.at("id").get<int>() != specie_id) {
                continue;
            }
            const int size_this_gen = specie.at("size_this_gen").get<int>();
            const float adjusted_fitness = size_this_gen > 0
                ? fitness / static_cast<float>(size_this_gen)
                : 0.0f;
            log.adjusted_fitness = adjusted_fitness;
            specie["adjusted_fitness_sum"] =
                specie.value("adjusted_fitness_sum", 0.0f) + adjusted_fitness;
            break;
        }
    }
    for (auto& specie : world.at("species")) {
        const int specie_id = specie.at("id").get<int>();
        const auto sum_it = score_factor_sums.find(specie_id);
        const int count = score_factor_counts[specie_id];
        const float inv = (sum_it != score_factor_sums.end() && count > 0)
            ? 1.0f / static_cast<float>(count) : 0.0f;
        const ScoreFactors& s = sum_it != score_factor_sums.end() ? sum_it->second : ScoreFactors{};
        specie["milliseconds_per_tick"] = s.milliseconds_per_tick * inv;
        specie["mean_spike_density"] = s.mean_spike_density * inv;
        specie["max_spike_density"] = s.max_spike_density * inv;
        specie["inhibitory_excitatory_ratio"] = s.inhibitory_excitatory_ratio * inv;
        specie["snn_neuron_synapse_ratio"] = s.snn_neuron_synapse_ratio * inv;
        specie["mean_last_click_prediction_error"] = s.mean_last_click_prediction_error * inv;
        specie["consistency_last_click_prediction_error"] = s.consistency_last_click_prediction_error * inv;
        specie["mean_last_click_left_hits"] = s.mean_last_click_left_hits * inv;
        specie["mean_last_click_right_hits"] = s.mean_last_click_right_hits * inv;
        specie["mean_click_type_decisiveness"] = s.mean_click_type_decisiveness * inv;
        specie["mean_certainty"] = s.mean_certainty * inv;
        specie["calibration_error"] = s.calibration_error * inv;
        specie["soft_jitter"] = s.soft_jitter * inv;
        specie["medium_jitter"] = s.medium_jitter * inv;
        specie["hard_jitter"] = s.hard_jitter * inv;
        specie["learning"] = s.learning * inv;
    }

    int active_species = 0;
    for (const auto& specie : world.at("species")) {
        if (specie.value("size_this_gen", 0) > 0) {
            ++active_species;
        }
    }
    if (active_species < speciation::k_target_min_species) {
        const float threshold = world["speciation"].value("compatibility_threshold", speciation::k_compatibility_threshold);
        world["speciation"]["compatibility_threshold"] = std::max(speciation::k_threshold_absolute_min, threshold - speciation::k_threshold_lower_step);
    } else if (active_species > speciation::k_target_max_species) {
        const float threshold = world["speciation"].value("compatibility_threshold", speciation::k_compatibility_threshold);
        world["speciation"]["compatibility_threshold"] = std::min(speciation::k_threshold_absolute_max, threshold + speciation::k_threshold_raise_step);
    }

    Util::JsonAtomicWrite(world_path, world);
    //return
    return organisms;
}

void Speciation::saveWorldManager()
{
    std::filesystem::path world_path =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    const nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);
    const int generation = world.at("generation").get<int>();

    const std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream date_stamp;
    date_stamp << std::put_time(&local_tm, "%Y-%m-%d_%H-%M-%S");

    const std::filesystem::path training_dir =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "records/training";
    std::error_code ec;
    std::filesystem::create_directories(training_dir, ec);
    if (ec) {
        Util::Errors::stop(
            std::string("Speciation: failed to create training archive directory: ")
            + training_dir.string()
            + ": "
            + ec.message());
    }

    const std::filesystem::path archive_path = training_dir / (
        std::to_string(generation) + "-generation-" + date_stamp.str() + ".json");

    Util::JsonAtomicWrite(archive_path, world);

    Util::Logs::write(
        "Speciation: saved world manager archive to "
        + archive_path.generic_string()
        + '\n');
}

void Speciation::pruneDeadSpecies()
{
    std::filesystem::path world_path =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    nlohmann::json remaining = nlohmann::json::array();
    for (const auto& specie : world.at("species")) {
        if (specie.value("size_this_gen", 0) > 0) {
            remaining.push_back(specie);
        }
    }
    const int pruned = static_cast<int>(world.at("species").size()) - static_cast<int>(remaining.size());
    world["species"] = std::move(remaining);

    Util::JsonAtomicWrite(world_path, world);

    if (pruned > 0) {
        Util::Logs::write(
            "Speciation: pruned " + std::to_string(pruned) + " dead species\n");
    }
}

void Speciation::updateWorldManager()
{
    if (g_organism_id_world_loaded) {
        Util::JsonAtomicWrite(g_organism_id_world_path, g_organism_id_world);
        g_organism_id_world_loaded = false;
    }

    std::filesystem::path world_path =
        std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    for (auto& specie : world.at("species")) {
        specie["best_fitness_this_gen"] = 0.0f;
        specie["size_this_gen"] = 0;
        specie["adjusted_fitness_sum"] = 0.0f;
        specie["milliseconds_per_tick"] = 0.0f;
        specie["mean_spike_density"] = 0.0f;
        specie["max_spike_density"] = 0.0f;
        specie["inhibitory_excitatory_ratio"] = 0.0f;
        specie["snn_neuron_synapse_ratio"] = 0.0f;
        specie["mean_last_click_prediction_error"] = 0.0f;
        specie["consistency_last_click_prediction_error"] = 0.0f;
        specie["mean_last_click_left_hits"] = 0.0f;
        specie["mean_last_click_right_hits"] = 0.0f;
        specie["mean_click_type_decisiveness"] = 0.0f;
        specie["mean_certainty"] = 0.0f;
        specie["calibration_error"] = 0.0f;
        specie["soft_jitter"] = 0.0f;
        specie["medium_jitter"] = 0.0f;
        specie["hard_jitter"] = 0.0f;
        specie["learning"] = 0.0f;
        specie["specie_differentiation"] = 0.0f;
        specie["differentiation_samples"] = 0;
    }
    world["generation"] = world.value("generation", 0) + 1;
    Util::JsonAtomicWrite(world_path, world);

}

Specie Speciation::getSpecie(int specie_id){
    //get specie json
    std::filesystem::path world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    for (const auto& specie : world.at("species")) {
        if (specie.at("id").get<int>() != specie_id) {
            continue;
        }

        //transfomr json into Specie struct
        Specie result{};
        result.id = specie.at("id").get<int32_t>();
        result.rep = specie.at("rep_path").get<std::string>();
        result.rep_id = specie.value("rep_id", 0);
        result.rep_fitness = specie.value("rep_fitness", 0.0f);
        result.created_generation = specie.value("created_generation", 0);
        result.size_this_gen = specie.value("size_this_gen", 0);
        result.best_fitness_this_gen = specie.value("best_fitness_this_gen", 0.0f);
        result.best_fitness_ever = specie.value("best_fitness_ever", 0.0f);
        result.stagnation_generations = specie.value("stagnation_generations", 0);
        result.offspring_count = specie.value("offspring_count", 0);
        ScoreFactors& sf = result.score_factors;
        sf.milliseconds_per_tick = specie.value("milliseconds_per_tick", 0.0f);
        sf.mean_spike_density = specie.value("mean_spike_density", 0.0f);
        sf.max_spike_density = specie.value("max_spike_density", 0.0f);
        sf.inhibitory_excitatory_ratio = specie.value("inhibitory_excitatory_ratio", 0.0f);
        sf.snn_neuron_synapse_ratio = specie.value("snn_neuron_synapse_ratio", 0.0f);
        sf.mean_last_click_prediction_error = specie.value("mean_last_click_prediction_error", 0.0f);
        sf.consistency_last_click_prediction_error =
            specie.value("consistency_last_click_prediction_error", 0.0f);
        sf.mean_last_click_left_hits = specie.value("mean_last_click_left_hits", 0.0f);
        sf.mean_last_click_right_hits = specie.value("mean_last_click_right_hits", 0.0f);
        sf.mean_click_type_decisiveness = specie.value("mean_click_type_decisiveness", 0.0f);
        sf.mean_certainty = specie.value("mean_certainty", 0.0f);
        sf.calibration_error = specie.value("calibration_error", 0.0f);
        sf.soft_jitter = specie.value("soft_jitter", 0.0f);
        sf.medium_jitter = specie.value("medium_jitter", 0.0f);
        sf.hard_jitter = specie.value("hard_jitter", 0.0f);
        sf.learning = specie.value("learning", 0.0f);
        //return Specie struct
        return result;
    }

    Util::Errors::stop("Speciation: specie id not found: " + std::to_string(specie_id));
}

void Speciation::updateOffspringShare(){

    if (g_organism_id_world_loaded) {
        Util::JsonAtomicWrite(g_organism_id_world_path, g_organism_id_world);
        g_organism_id_world_loaded = false;
    }

    std::filesystem::path world_path = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json";
    nlohmann::json world = Util::JsonAtomicOpen(world_path, &world_path);

    const int target_population = world.at("population_size").get<int>();
    const int stagnation_limit = world.value("stagnation_limit", speciation::k_stagnation_limit);

    std::vector<std::size_t> non_stagnant_indices;
    non_stagnant_indices.reserve(world.at("species").size());
    float total_adjusted = 0.0f;

    for (std::size_t i = 0; i < world.at("species").size(); ++i) {
        auto& specie = world.at("species")[i];
        // stagnant species get 0 only while other species are still reproducing
        if (specie.value("stagnation_generations", 0) > stagnation_limit
            || specie.value("size_this_gen", 0) <= 0) {
            specie["offspring_count"] = 0;
            continue;
        }
        non_stagnant_indices.push_back(i);
        total_adjusted += specie.value("adjusted_fitness_sum", 0.0f);
    }
    // all species stagnated simultaneously: give every live species one reprieve generation
    // (reset to stagnation_limit, not 0) so the floor fires again next gen if they don't improve
    if (non_stagnant_indices.empty()) {
        non_stagnant_indices.clear();
        total_adjusted = 0.0f;
        for (std::size_t i = 0; i < world.at("species").size(); ++i) {
            auto& specie = world.at("species")[i];
            if (specie.value("size_this_gen", 0) <= 0) continue;
            specie["stagnation_generations"] = stagnation_limit;
            non_stagnant_indices.push_back(i);
            total_adjusted += specie.value("adjusted_fitness_sum", 0.0f);
        }
    }

    std::vector<int> offspring_counts(non_stagnant_indices.size(), 0);

    if (!non_stagnant_indices.empty()) {
        // Edge case: if every adjusted sum is 0, fall back to equal split among non-stagnant species.
        if (total_adjusted <= 0.0f) {
            const int base = target_population / static_cast<int>(non_stagnant_indices.size());
            const int remainder = target_population % static_cast<int>(non_stagnant_indices.size());
            for (std::size_t i = 0; i < non_stagnant_indices.size(); ++i) {
                offspring_counts[i] = base + (static_cast<int>(i) < remainder ? 1 : 0);
            }
        } else {
            // -Allocate non-stagnant the offspring proportionally by adjusted fitness sum
            int allocated = 0;
            for (std::size_t i = 0; i < non_stagnant_indices.size(); ++i) {
                const auto& specie = world.at("species")[non_stagnant_indices[i]];
                // offspring_for_species = round(population_size * sum(adjusted_score_in_species) / sum(adjusted_score_in_all)))
                if (i + 1 == non_stagnant_indices.size()) {
                    // for last non-stagnant just do  total - sum(offspring_for_species) for rounding errors
                    offspring_counts[i] = std::max(1, target_population - allocated);
                } else {
                    const float specie_sum = specie.value("adjusted_fitness_sum", 0.0f);
                    offspring_counts[i] = std::max(1, static_cast<int>(std::round(
                        static_cast<float>(target_population) * specie_sum / total_adjusted)));
                    allocated += offspring_counts[i];
                }
            }
        }
    }

    // check if any offspring share is less than 0, if so make share 1 and population_size--
    for (int& share : offspring_counts) {
        if (share < 0) {
            share = 1;
        }
    }
    int allocated_sum = 0;
    for (int share : offspring_counts) {
        allocated_sum += share;
    }
    if (!offspring_counts.empty() && allocated_sum != target_population) {
        offspring_counts.back() += target_population - allocated_sum;
    }

    for (std::size_t i = 0; i < non_stagnant_indices.size(); ++i) {
        world.at("species")[non_stagnant_indices[i]]["offspring_count"] = offspring_counts[i];
    }

    world["engine"]["train_tick_count"] = neat_pipeline::k_train_tick_count;
    world["engine"]["tick_target_ms"] = static_cast<long long>(runtime::k_tick_target.count());
    world["engine"]["stimulus_speed"] = fitness_evaluation::k_stimulus_speed;
    world["engine"]["direction_dot_threshold"] = fitness_evaluation::k_direction_dot_threshold;
    world["engine"]["response_prediction_motion"] = fitness_evaluation::k_response_prediction_motion;
    world["engine"]["jitter_distance_norm"] = fitness_evaluation::k_jitter_distance_norm;
    world["engine"]["target_mean_spike_density"] = fitness_evaluation::k_target_mean_spike_density;

    world["speciation"]["output_log_tick_interval"] = speciation::k_output_log_tick_interval;
    world["speciation"]["behavior_position_weight"] = speciation::k_behavior_position_weight;
    world["speciation"]["behavior_click_weight"] = speciation::k_behavior_click_weight;
    world["speciation"]["target_min_species"] = speciation::k_target_min_species;
    world["speciation"]["target_max_species"] = speciation::k_target_max_species;
    world["speciation"]["threshold_raise_step"] = speciation::k_threshold_raise_step;
    world["speciation"]["threshold_lower_step"] = speciation::k_threshold_lower_step;
    world["speciation"]["threshold_absolute_min"] = speciation::k_threshold_absolute_min;
    world["speciation"]["threshold_absolute_max"] = speciation::k_threshold_absolute_max;

    world["mutation"]["decimal_mutation_mod_min"] = mutation::k_decimal_mutation_mod_min;
    world["mutation"]["decimal_mutation_mod_max"] = mutation::k_decimal_mutation_mod_max;
    world["mutation"]["integer_mutation_mod_min"] = mutation::k_integer_mutation_mod_min;
    world["mutation"]["integer_mutation_mod_max"] = mutation::k_integer_mutation_mod_max;
    world["mutation"]["global_genome_mutation_chance"] = mutation::k_global_genome_mutation_chance;
    world["mutation"]["global_genome_random_change_chance"] = mutation::k_global_genome_random_change_chance;
    world["mutation"]["hidden_neuron_mutation_chance"] = mutation::k_hidden_neuron_mutation_chance;
    world["mutation"]["hidden_neuron_random_change_chance"] = mutation::k_hidden_neuron_random_change_chance;
    world["mutation"]["output_neuron_mutation_chance"] = mutation::k_output_neuron_mutation_chance;
    world["mutation"]["output_neuron_random_change_chance"] = mutation::k_output_neuron_random_change_chance;
    world["mutation"]["bias_neuron_mutation_chance"] = mutation::k_bias_neuron_mutation_chance;
    world["mutation"]["bias_neuron_random_change_chance"] = mutation::k_bias_neuron_random_change_chance;
    world["mutation"]["input_neuron_mutation_chance"] = mutation::k_input_neuron_mutation_chance;
    world["mutation"]["input_neuron_random_change_chance"] = mutation::k_input_neuron_random_change_chance;
    world["mutation"]["connection_genome_mutation_chance"] = mutation::k_connection_genome_mutation_chance;
    world["mutation"]["connection_genome_random_change_chance"] = mutation::k_connection_genome_random_change_chance;
    world["mutation"]["new_connection_genome_chance"] = mutation::k_new_connection_genome_chance;
    world["mutation"]["disable_connection_genome_chance"] = mutation::k_disable_connection_genome_chance;
    world["mutation"]["enable_connection_genome_chance"] = mutation::k_enable_connection_genome_chance;
    world["mutation"]["new_neuron_genome_chance"] = mutation::k_new_neuron_genome_chance;

    //update specie offspring_count on json
    Util::JsonAtomicWrite(world_path, world);
}
