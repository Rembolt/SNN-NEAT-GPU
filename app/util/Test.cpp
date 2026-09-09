#include "util/Log.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// mirrors speciation_json::saveToPath in Speciation.cpp
void saveJsonLikeSpeciation(const std::filesystem::path& path, const nlohmann::json& json)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        Util::Errors::stop("Test: failed to open for write: " + path.string());
    }
    out << json.dump(2);
}

nlohmann::json loadJsonFile(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        Util::Errors::stop("Test: failed to open for read: " + path.string());
    }
    return nlohmann::json::parse(in);
}

std::filesystem::path testWorkspace()
{
    std::error_code ec;
    const std::filesystem::path workspace =
        std::filesystem::temp_directory_path(ec) / "snn_gpu_speciation_file_tests";
    if (ec) {
        Util::Errors::stop("Test: temp_directory_path failed");
    }
    std::filesystem::create_directories(workspace, ec);
    if (ec) {
        Util::Errors::stop("Test: failed to create workspace: " + workspace.string());
    }
    return workspace;
}

std::filesystem::path sourceAdamPath()
{
    return std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "DNAs" / "Adam.json";
}

bool testRoundTripPreservesGenomeSections()
{
    const std::filesystem::path workspace = testWorkspace();
    const std::filesystem::path source = sourceAdamPath();
    const std::filesystem::path copy_path = workspace / "round_trip.json";

    std::error_code ec;
    std::filesystem::copy_file(
        source,
        copy_path,
        std::filesystem::copy_options::overwrite_existing,
        ec);
    if (ec) {
        Util::Logs::write("Test FAIL round_trip: could not copy Adam.json: " + ec.message() + "\n");
        return false;
    }

    const nlohmann::json before = loadJsonFile(copy_path);
    const std::size_t neuron_count_before = before.at("neuron_genomes").size();
    const std::size_t connection_count_before = before.at("connection_genomes").size();

    nlohmann::json organism = before;
    organism.at("global_genome")["fitness"] = 123.456f;
    saveJsonLikeSpeciation(copy_path, organism);

    const nlohmann::json after = loadJsonFile(copy_path);
    if (!after.contains("global_genome") || !after.contains("neuron_genomes") || !after.contains("connection_genomes")) {
        Util::Logs::write("Test FAIL round_trip: top-level genome sections missing after save\n");
        return false;
    }
    if (after.at("neuron_genomes").size() != neuron_count_before) {
        Util::Logs::write("Test FAIL round_trip: neuron_genomes count changed\n");
        return false;
    }
    if (after.at("connection_genomes").size() != connection_count_before) {
        Util::Logs::write("Test FAIL round_trip: connection_genomes count changed\n");
        return false;
    }
    if (after.at("global_genome").at("fitness").get<float>() != 123.456f) {
        Util::Logs::write("Test FAIL round_trip: fitness field not updated\n");
        return false;
    }

    Util::Logs::write("Test PASS round_trip: save/load preserves all genome sections\n");
    return true;
}

bool testCopyFileMatchesSourceBytes()
{
    const std::filesystem::path workspace = testWorkspace();
    const std::filesystem::path source = sourceAdamPath();
    const std::filesystem::path dest = workspace / "copied.json";

    std::error_code ec;
    std::filesystem::copy_file(
        source,
        dest,
        std::filesystem::copy_options::overwrite_existing,
        ec);
    if (ec) {
        Util::Logs::write("Test FAIL copy_bytes: copy_file failed: " + ec.message() + "\n");
        return false;
    }

    const std::uintmax_t source_size = std::filesystem::file_size(source, ec);
    const std::uintmax_t dest_size = std::filesystem::file_size(dest, ec);
    if (ec || source_size != dest_size) {
        Util::Logs::write("Test FAIL copy_bytes: copied file size mismatch\n");
        return false;
    }

    const nlohmann::json copied = loadJsonFile(dest);
    if (copied.at("neuron_genomes").empty()) {
        Util::Logs::write("Test FAIL copy_bytes: copied file has empty neuron_genomes\n");
        return false;
    }

    Util::Logs::write("Test PASS copy_bytes: copy_file keeps full organism JSON\n");
    return true;
}

bool testStaleReadModifyWriteLosesUpdates()
{
    const std::filesystem::path workspace = testWorkspace();
    const std::filesystem::path world_path = workspace / "world-manager-stale.json";

    nlohmann::json initial = {
        {"next_organism_id", 2},
        {"species", nlohmann::json::array()},
    };
    saveJsonLikeSpeciation(world_path, initial);

    // two callers load the same counter before either saves (stale snapshot pattern)
    nlohmann::json snapshot_a = loadJsonFile(world_path);
    nlohmann::json snapshot_b = loadJsonFile(world_path);

    const int id_a = snapshot_a.at("next_organism_id").get<int>();
    snapshot_a["next_organism_id"] = id_a + 1;
    saveJsonLikeSpeciation(world_path, snapshot_a);

    const int id_b = snapshot_b.at("next_organism_id").get<int>();
    snapshot_b["next_organism_id"] = id_b + 1;
    saveJsonLikeSpeciation(world_path, snapshot_b);

    const int final_id = loadJsonFile(world_path).at("next_organism_id").get<int>();
    if (final_id == 3) {
        Util::Logs::write(
            "Test PASS stale_rmw: stale snapshot dropped an increment (final next_organism_id=3, expected 4)\n");
        return true;
    }

    Util::Logs::write(
        "Test FAIL stale_rmw: unexpected final next_organism_id="
        + std::to_string(final_id)
        + " (expected 3 from stale clobber)\n");
    return false;
}

bool testTruncateBeforeWriteCanLeavePartialJson()
{
    const std::filesystem::path workspace = testWorkspace();
    const std::filesystem::path source = sourceAdamPath();
    const std::filesystem::path victim = workspace / "truncated.json";

    std::error_code ec;
    std::filesystem::copy_file(
        source,
        victim,
        std::filesystem::copy_options::overwrite_existing,
        ec);
    if (ec) {
        Util::Logs::write("Test FAIL truncate: setup copy failed\n");
        return false;
    }

    const nlohmann::json full = loadJsonFile(victim);
    const std::string payload = full.dump(2);

    // ofstream truncates immediately; simulate an interrupted rewrite
    {
        std::ofstream out(victim, std::ios::trunc);
        out.write(payload.data(), static_cast<std::streamsize>(payload.size() / 4));
    }

    std::ifstream in(victim);
    bool parse_failed = false;
    try {
        nlohmann::json::parse(in);
    } catch (const nlohmann::json::parse_error&) {
        parse_failed = true;
    }

    if (!parse_failed) {
        Util::Logs::write("Test FAIL truncate: partial write still parsed as valid JSON\n");
        return false;
    }

    Util::Logs::write("Test PASS truncate: non-atomic rewrite can corrupt organism JSON\n");
    return true;
}

}  // namespace

bool runSpeciationFileTests()
{
    Util::Logs::write("Test: running speciation file I/O checks\n");

    const std::vector<bool> results = {
        testRoundTripPreservesGenomeSections(),
        testCopyFileMatchesSourceBytes(),
        testStaleReadModifyWriteLosesUpdates(),
        testTruncateBeforeWriteCanLeavePartialJson(),
    };

    for (bool ok : results) {
        if (!ok) {
            Util::Logs::write("Test: one or more checks failed\n");
            return false;
        }
    }

    Util::Logs::write("Test: all speciation file I/O checks passed\n");
    return true;
}
