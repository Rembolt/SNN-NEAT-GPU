#include "NeatPipeline.h"
#include "Constants.h"
#include "Reproduction.h"

#include "util/Log.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>


void FitnessScoreCallback(
    void* user_data,
    unsigned long tick_index,
    std::uint32_t spiked_only_count,
    double tick_elapsed_ms,
    double tick_target_ms,
    bool mouse_click_edge,
    const HostFeedbackInput& feedback,
    const HostMouseOutput& mouse_out)
{
    static_cast<FitnessEvaluation*>(user_data)->updateScoreFactorsAtRuntime(
        tick_index,
        spiked_only_count,
        tick_elapsed_ms,
        tick_target_ms,
        mouse_click_edge,
        feedback,
        mouse_out);
}

std::vector<OrganismLog> averageLogs(
    std::vector<OrganismLog> list_a,
    std::vector<OrganismLog> list_b)
{

    std::sort(
        list_a.begin(),
        list_a.end(),
        [](const OrganismLog& a, const OrganismLog& b) { return a.organism_path < b.organism_path; }
    );

    std::sort(
        list_b.begin(),
        list_b.end(),
        [](const OrganismLog& a, const OrganismLog& b) { return a.organism_path < b.organism_path; }
    );
    if (list_a.size() != list_b.size()) {
        Util::Errors::stop("averageOrganismLogsByPath: score list size mismatch");
    }

    std::vector<OrganismLog> averaged;
    averaged.reserve(list_a.size());
    for (std::size_t i = 0; i < list_a.size(); ++i) {
        if (list_a[i].organism_path != list_b[i].organism_path) {
            Util::Errors::stop("averageOrganismLogsByPath: dna path mismatch");
        }
        const float avg_score =
            (list_a[i].raw_fitness + list_b[i].raw_fitness) / 2.0f;
        ScoreFactors avg_factors{};
        const float* a = reinterpret_cast<const float*>(&list_a[i].score_factors);
        const float* b = reinterpret_cast<const float*>(&list_b[i].score_factors);
        float* out = reinterpret_cast<float*>(&avg_factors);
        constexpr std::size_t k_factor_count = sizeof(ScoreFactors) / sizeof(float);
        for (std::size_t f = 0; f < k_factor_count; ++f) {
            out[f] = (a[f] + b[f]) / 2.0f;
        }
        averaged.push_back(OrganismLog{
            list_a[i].organism_path,
            list_a[i].organism_id,
            list_a[i].generation,
            list_a[i].specie_id,
            avg_score,
            0.0f,
            avg_factors,
            list_a[i].output_log});
    }
    return averaged;
}

void printOrganismLog(const OrganismLog& log)
{
    std::ostringstream line;
    line << "organism=" << log.organism_path
         << " id=" << log.organism_id
         << " specie_id=" << log.specie_id
         << " score=" << log.raw_fitness
         << " ms_per_tick=" << log.score_factors.milliseconds_per_tick
         << " mean_spike_density=" << log.score_factors.mean_spike_density
         << " mean_click_err=" << log.score_factors.mean_last_click_prediction_error
         << '\n';

    Util::Logs::write(line.str());
}

std::vector<std::filesystem::path> NeatPipeline::listMouseInputClipFiles(std::string_view subfolder)

{
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    fs::path clips_dir = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / mouse_clip::k_dir_name;
    if (!subfolder.empty()) {
        clips_dir /= subfolder;
    }

    if (!fs::is_directory(clips_dir)) {
        return paths;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(clips_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == mouse_clip::k_extension) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

void NeatPipeline::saveMouseInputClip(const std::vector<HostMouseInput>& input_history)
{
    namespace fs = std::filesystem;
    static_assert(std::is_trivially_copyable_v<HostMouseInput>);

    if (input_history.empty()) {
        Util::Errors::stop("saveMouseInputClip: input_history is empty");
    }

    const fs::path clips_dir = std::filesystem::path(SNN_GPU_PROJECT_ROOT) / mouse_clip::k_dir_name;
    std::error_code ec;
    fs::create_directories(clips_dir, ec);
    if (ec) {
        Util::Errors::stop(
            "saveMouseInputClip: failed to create directory "
            + clips_dir.string()
            + ": "
            + ec.message());
    }

    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const fs::path clip_path =
        clips_dir / ("clip_" + std::to_string(now_ms) + mouse_clip::k_extension);
    const fs::path temp_path = clip_path.string() + ".tmp";

    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        Util::Errors::stop("saveMouseInputClip: failed to open " + temp_path.string());
    }

    const std::uint32_t version = mouse_clip::k_version;
    const std::uint64_t tick_count = input_history.size();
    const std::uint32_t struct_size = static_cast<std::uint32_t>(sizeof(HostMouseInput));

    out.write(reinterpret_cast<const char*>(&mouse_clip::k_magic), sizeof(mouse_clip::k_magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&struct_size), sizeof(struct_size));
    out.write(reinterpret_cast<const char*>(&tick_count), sizeof(tick_count));
    out.write(
        reinterpret_cast<const char*>(input_history.data()),
        static_cast<std::streamsize>(input_history.size() * sizeof(HostMouseInput)));

    if (!out) {
        Util::Errors::stop("saveMouseInputClip: write failed for " + temp_path.string());
    }
    out.close();
    Util::atomicFileReplace(temp_path, clip_path);

    Util::Logs::write(
        "NeatPipeline: saved mouse input clip "
        + clip_path.filename().string()
        + " ("
        + std::to_string(tick_count)
        + " ticks)\n");
}

std::vector<HostMouseInput> NeatPipeline::loadMouseInputClip(
    const std::filesystem::path& clip_path,
    unsigned long expected_tick_count)
{
    static_assert(std::is_trivially_copyable_v<HostMouseInput>);

    std::ifstream in(clip_path, std::ios::binary);
    if (!in) {
        Util::Errors::stop("loadMouseInputClip: failed to open " + clip_path.string());
    }

    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (content.empty()) {
        Util::Errors::stop("loadMouseInputClip: empty file " + clip_path.string());
    }

    std::uint32_t magic = 0u;
    std::uint32_t version = 0u;
    std::uint32_t struct_size = 0u;
    std::uint64_t tick_count = 0u;

    constexpr std::size_t k_header_size =
        sizeof(magic) + sizeof(version) + sizeof(struct_size) + sizeof(tick_count);
    if (content.size() < k_header_size) {
        Util::Errors::stop("loadMouseInputClip: header read failed for " + clip_path.string());
    }

    std::size_t offset = 0;
    auto read_pod = [&](auto& field) {
        using field_type = std::decay_t<decltype(field)>;
        std::memcpy(&field, content.data() + offset, sizeof(field_type));
        offset += sizeof(field_type);
    };

    read_pod(magic);
    read_pod(version);
    read_pod(struct_size);
    read_pod(tick_count);
    if (magic != mouse_clip::k_magic) {
        Util::Errors::stop("loadMouseInputClip: invalid magic in " + clip_path.string());
    }
    if (version != mouse_clip::k_version) {
        Util::Errors::stop("loadMouseInputClip: unsupported version in " + clip_path.string());
    }
    if (struct_size != sizeof(HostMouseInput)) {
        Util::Errors::stop("loadMouseInputClip: HostMouseInput size mismatch in " + clip_path.string());
    }
    if (tick_count != expected_tick_count) {
        Util::Errors::stop(
            "loadMouseInputClip: tick_count ("
            + std::to_string(tick_count)
            + ") != expected ("
            + std::to_string(expected_tick_count)
            + ") in "
            + clip_path.string());
    }

    const std::size_t payload_size = static_cast<std::size_t>(tick_count) * sizeof(HostMouseInput);
    if (content.size() < offset + payload_size) {
        Util::Errors::stop("loadMouseInputClip: payload read failed for " + clip_path.string());
    }

    std::vector<HostMouseInput> input_history(static_cast<std::size_t>(tick_count));
    std::memcpy(input_history.data(), content.data() + offset, payload_size);

    return input_history;
}

std::vector<HostMouseInput> NeatPipeline::loadMouseInputClipFromFolder(
    App& app,
    unsigned long expected_tick_count)
{
    std::vector<std::filesystem::path> clip_files = listMouseInputClipFiles();
    if (clip_files.empty()) {
        Util::RecordClips::populateMouseInputClipFolder(
            app,
            neat_pipeline::k_train_tick_count,
            runtime::k_tick_target,
            neat_pipeline::k_clicks_required,
            mouse_clip::k_target_clip_count,
            neat_pipeline::k_record_attempts,
            [this](const std::vector<HostMouseInput>& input_history) {
                saveMouseInputClip(input_history);
            });
        clip_files = listMouseInputClipFiles();
    }
    if (clip_files.empty()) {
        Util::Errors::stop(
            "loadMouseInputClipFromFolder: no "
            + std::string(mouse_clip::k_extension)
            + " files in "
            + (std::filesystem::path(SNN_GPU_PROJECT_ROOT) / mouse_clip::k_dir_name).string());
    }

    //pick random clip from folder
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, clip_files.size() - 1);
    const std::filesystem::path clip_path = clip_files[dis(gen)];
    return loadMouseInputClip(clip_path, expected_tick_count);
}

std::vector<OrganismLog> NeatPipeline::loadDnaJsonFiles()
{
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    const fs::path dna_dir =
        fs::path(SNN_GPU_PROJECT_ROOT) / "DNAs";

    if (!fs::is_directory(dna_dir)) {
        Util::Errors::stop("NeatPipeline: DNAs directory not found: " + dna_dir.string());
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(dna_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        Util::Errors::stop("NeatPipeline: no .json files in DNAs/");
    }

    std::vector<OrganismLog> organism_logs;
    organism_logs.reserve(paths.size());
    for (const std::filesystem::path& path : paths) {
        nlohmann::json json = nlohmann::json::parse(std::ifstream(path));
        organism_logs.push_back(OrganismLog{
            path.string(),
            json.at("global_genome").at("organism_id").get<int>(),
            json.at("global_genome").at("generation").get<int>(),
            json.at("global_genome").at("species_id").get<int>(),
            json.at("global_genome").at("fitness").get<float>(),
            json.at("global_genome").at("adjusted_fitness").get<float>(),
            ScoreFactors{},
            OutputLog{}});
    }
    return organism_logs;
}


std::vector<HostMouseInput> NeatPipeline::getNewMouseInputClip(App& app) {
    Util::Logs::write(
        "NeatPipeline: recording mouse input for "
        + std::to_string(neat_pipeline::k_train_tick_count)
        + " ticks (~"
        + std::to_string(
              std::chrono::duration_cast<std::chrono::seconds>(
                  runtime::k_tick_target * neat_pipeline::k_train_tick_count)
                  .count())
        + "s at 16ms/tick). Move the mouse and click.\n"
    );
    
    //attempt to record a clip that has good amount of clicks for training
    std::vector<HostMouseInput> input_history;
    for(int i = 0; i < neat_pipeline::k_record_attempts; i++) {
        input_history = app.recordedInputForTraining(neat_pipeline::k_train_tick_count, runtime::k_tick_target, neat_pipeline::k_clicks_required);
        if(!input_history.empty()) {
            break;
        }
    }
    //pick from folder if no good clip found
    if(input_history.empty()) {
        input_history = loadMouseInputClipFromFolder(app, neat_pipeline::k_train_tick_count);
    }else{
        saveMouseInputClip(input_history);
    }

    return input_history;
}

std::vector<HostMouseInput> NeatPipeline::getExistingMouseInputClip(App& app) {
    Util::Logs::write(
        "NeatPipeline: utilizing previously recorded mouse input for training"
        
    );
    
    const std::vector<HostMouseInput> input_history =
        loadMouseInputClipFromFolder(app, neat_pipeline::k_train_tick_count);

    return input_history;
}

std::vector<OrganismLog> NeatPipeline::generationRun(
    const CLBoilerplate& cl,
    SNNEngine& snn_engine,
    App& app,
    const std::vector<HostMouseInput>& input_history,
    const std::vector<OrganismLog>* organism_logs)
{
    //evaluate all dnas in folder or all dnas in list given as parameter
    std::vector<std::filesystem::path> dna_files;
    dna_files.reserve(organism_logs->size());
    for (const OrganismLog& organism_log : *organism_logs) {
        dna_files.emplace_back(organism_log.organism_path.c_str());
    }

    std::vector<OrganismLog> scored_logs;
    scored_logs.reserve(organism_logs->size());

    for (std::size_t i = 0; i < dna_files.size(); ++i) {
        //debug organism number
        Util::Logs::write("NeatPipeline: running organism " + std::to_string(i));
        const std::filesystem::path& dna_path = dna_files[i];
        const OrganismLog& source_log = (*organism_logs)[i];
        DNAReader dna_reader;
        dna_reader.readDNAFile(dna_path.string());
        HostOrganismBuffers h_organism_buffers = dna_reader.getOrganismBuffers();
        HostAppParams h_app_params = dna_reader.getAppParams();

        app.setMode(AppMode::TrainReplay);
        app.allocateBuffers(
            cl.device(),
            cl.context(),
            cl.queue(),
            h_organism_buffers,
            h_app_params);
        app.initializeSnnKernels();
        app.resetOrganismState();

        FitnessEvaluation fitness_evaluation;
        fitness_evaluation.beginRun(
            neat_pipeline::k_train_tick_count,
            h_organism_buffers.h_global_params_.h_num_neurons,
            h_organism_buffers.h_global_params_.h_num_synapses);

        app.tickForTraining(
            input_history,
            neat_pipeline::k_train_tick_count,
            runtime::k_tick_target,
            FitnessScoreCallback,
            &fitness_evaluation);

        std::uint32_t final_inhibitory_spike_rate = 0u;
        std::uint32_t final_excitatory_spike_rate = 0u;
        if (!snn_engine.readPropagationSpikeRates(
                final_inhibitory_spike_rate,
                final_excitatory_spike_rate)) {
            Util::Errors::stop("NeatPipeline: readPropagationSpikeRates failed");
        }

        fitness_evaluation.calculateScoreFactors(
            final_inhibitory_spike_rate,
            final_excitatory_spike_rate);

        const int score = fitness_evaluation.calculateScore();
        const ScoreFactors& factors = fitness_evaluation.scoreFactors();


        scored_logs.push_back(OrganismLog{
            dna_path.string(),
            source_log.organism_id,
            source_log.generation,
            source_log.specie_id,
            static_cast<float>(score),
            0.0f,
            factors,
            fitness_evaluation.outputLog()});

        app.freeBuffers();
    }
    
    return scored_logs;
}


void NeatPipeline::safeStart()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(SNN_GPU_PROJECT_ROOT);
    const fs::path committed_world_path = root / "species/commited-world-manager.json";
    const fs::path committed_innovation_path = root / "app/evolution/commited-innovation-ids.json";
    const fs::path world_path = root / "species/world-manager.json";
    const fs::path innovation_path = root / "app/evolution/innovation_ids.json";
    const fs::path dna_dir = root / "DNAs";
    const fs::path species_dir = root / "species";
    const fs::path clips_dir = root / mouse_clip::k_dir_name;
    const fs::path evolution_dir = root / "app/evolution";


    //check for corruption in committed world and innovation ids and write committed to current generation file
    if (!fs::exists(committed_world_path)) {
        if (Util::isWorldCurrupted(world_path)) {
            Util::Errors::stop("safeStart: missing committed world and live world is corrupted");
        }
        Util::JsonAtomicWrite(committed_world_path, Util::JsonAtomicOpen(world_path));
    }
    if (!fs::exists(committed_innovation_path)) {
        if (Util::areInnovationIdsCurrupted(innovation_path)) {
            Util::Errors::stop("safeStart: missing committed innovation ids and live file is corrupted");
        }
        Util::JsonAtomicWrite(committed_innovation_path, Util::JsonAtomicOpen(innovation_path));
    }
    if (Util::isWorldCurrupted(committed_world_path)) {
        Util::Errors::stop("safeStart: corrupted " + committed_world_path.string());
    }
    if (Util::areInnovationIdsCurrupted(committed_innovation_path)) {
        Util::Errors::stop("safeStart: corrupted " + committed_innovation_path.string());
    }

    nlohmann::json world = Util::JsonAtomicOpen(committed_world_path);
    const int current_generation = world.at("generation").get<int>();

    int max_organism_id = -1;
    if (fs::is_directory(dna_dir)) {
        for (const fs::directory_entry& entry : fs::directory_iterator(dna_dir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                continue;
            }
            const fs::path& path = entry.path();
            //check for corruption in dna files
            if (Util::isDNACurrupted(path)) {
                Util::Errors::stop("safeStart: corrupted " + path.string());
            }
            const nlohmann::json dna = Util::JsonAtomicOpen(path);
            const nlohmann::json& global = dna.at("global_genome");
            //delete dna files of past or unfinished generation
            if (global.at("generation").get<int>() != current_generation) {
                std::error_code ec;
                fs::remove(path, ec);
                if (ec) {
                    Util::Errors::stop("safeStart: failed to delete " + path.string());
                }
                continue;
            }
            max_organism_id = std::max(max_organism_id, global.at("organism_id").get<int>());
        }
    }
    //clean stray files (.tmp, half-written clips, stale locks)
    const fs::path cleanup_dirs[] = {dna_dir, species_dir, clips_dir, evolution_dir};
    for (const fs::path& dir : cleanup_dirs) {
        if (!fs::is_directory(dir)) {
            continue;
        }
        for (const fs::directory_entry& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".tmp") {
                std::error_code ec;
                fs::remove(entry.path(), ec);
            }
        }
    }

    //check for corruption in species rep and legend files
    for (auto& specie : world.at("species")) {
        const int specie_id = specie.at("id").get<int>();
        for (const char* suffix : {"rep", "legend"}) {
            const fs::path path = species_dir / (std::to_string(specie_id) + "-" + suffix + ".json");
            if (Util::isDNACurrupted(path)) {
                Util::Errors::stop("safeStart: corrupted species file " + path.string());
            }
        }
    }
    //check for corruption in input clips
    for (const fs::path& clip_path : listMouseInputClipFiles()) {
        if (Util::isClipCurrupted(clip_path)) {
            Util::Errors::stop("safeStart: corrupted clip " + clip_path.string());
        }
    }

    world["next_organism_id"] = max_organism_id + 1;
    Util::JsonAtomicWrite(world_path, world);
    Util::JsonAtomicWrite(innovation_path, Util::JsonAtomicOpen(committed_innovation_path));
}

void NeatPipeline::progressTest(
    const CLBoilerplate& cl,
    SNNEngine& snn_engine,
    App& app,
    const std::map<int, std::vector<OrganismLog>>& specie_organisms)
{
    //top organisms of every specie are the only candidates for the generalization test
    std::vector<OrganismLog> candidates;
    for (const auto& [specie_id, organisms] : specie_organisms) {
        const std::size_t top_count = std::min(organisms.size(), neat_pipeline::k_progress_test_top_organisms);
        candidates.insert(candidates.end(), organisms.begin(), organisms.begin() + top_count);
    }

    std::vector<std::filesystem::path> clip_files =
        listMouseInputClipFiles(mouse_clip::k_progress_test_dir_name);
    if (clip_files.size() > neat_pipeline::k_progress_test_max_clips) {
        clip_files.resize(neat_pipeline::k_progress_test_max_clips);
    }
    if (candidates.empty() || clip_files.empty()) {
        Util::Logs::write("NeatPipeline: progress test skipped, no candidates or no progress-test clips\n");
        return;
    }

    //score every candidate on every progress-test clip, generationRun keeps the input order
    std::vector<float> score_sums(candidates.size(), 0.0f);
    for (const std::filesystem::path& clip_path : clip_files) {
        const std::vector<HostMouseInput> input_clip =
            loadMouseInputClip(clip_path, neat_pipeline::k_train_tick_count);
        const std::vector<OrganismLog> clip_scores =
            generationRun(cl, snn_engine, app, input_clip, &candidates);
        for (std::size_t i = 0; i < score_sums.size(); ++i) {
            score_sums[i] += clip_scores[i].raw_fitness;
        }
        Util::Logs::write(
            "NeatPipeline: progress test ran clip " + clip_path.filename().string() + '\n');
    }

    //mean across clips is the generalization score handed to speciation
    const float inv_clip_count = 1.0f / static_cast<float>(clip_files.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].raw_fitness = score_sums[i] * inv_clip_count;
        Util::Logs::write(
            "NeatPipeline: progress test organism " + candidates[i].organism_path
            + " specie " + std::to_string(candidates[i].specie_id)
            + " mean score " + std::to_string(candidates[i].raw_fitness) + '\n');
    }

    Speciation::progressTest(candidates);
}

void NeatPipeline::run()
{
    CLBoilerplate cl;
    cl.printDevice();

    SNNEngine snn_engine;
    App app(snn_engine);
    app.setMode(AppMode::RecordInput);
    app.initializeInputCapture();

    //loop until shutdown
    while (true) {
        generation_progress_percentage = 0;
        safeStart();
        generation_progress_percentage = 5;
        Util::Logs::write(
            "generation_progress_percentage=" + std::to_string(generation_progress_percentage)
            + "% SAFE to shutdown\n");

        std::vector<OrganismLog> organism_logs = loadDnaJsonFiles();
        //debug log
        Util::Logs::write("NeatPipeline: loaded " + std::to_string(organism_logs.size()) + " organisms");

        // test generation on new clip
        std::vector<HostMouseInput> input_clip = getNewMouseInputClip(app);
        // // test generation on old clip
        // std::vector<HostMouseInput> input_clip = getExistingMouseInputClip(app);

        //debug log
        Util::Logs::write("NeatPipeline: got input clip");
        std::vector<OrganismLog> refined_scores = generationRun(cl, snn_engine, app, input_clip, &organism_logs);
        //speciation compares behaviour, so reps need a trace from this same clip
        const std::vector<OrganismLog> reps = Speciation::repOrganisms();
        std::vector<OrganismLog> rep_logs;
        if (!reps.empty()) {
            rep_logs = generationRun(cl, snn_engine, app, input_clip, &reps);
        }
        generation_progress_percentage = 60;
        Util::Logs::write(
            "generation_progress_percentage=" + std::to_string(generation_progress_percentage)
            + "% SAFE to shutdown\n");
        //debug log
        Util::Logs::write("NeatPipeline: ran generation 1");
       
        // //RUN SECOND TIME:
        // //sort in descending order so the split below actually picks the best scorers
        // std::sort(
        //     initial_scores_run1.begin(),
        //     initial_scores_run1.end(),
        //     [](const OrganismLog& a, const OrganismLog& b) { return a.raw_fitness > b.raw_fitness; }
        // );
        // //split organisms into top and bottom
        // const std::size_t number_of_top_organisms = std::max(
        //     std::size_t{1},
        //     static_cast<std::size_t>(std::ceil(initial_scores_run1.size() * neat_pipeline::k_percentage_of_organism_to_test_more_times))
        // );

        // std::vector<OrganismLog> bottom_organisms(
        //     std::make_move_iterator(initial_scores_run1.begin() + number_of_top_organisms),
        //     std::make_move_iterator(initial_scores_run1.end()));
        // initial_scores_run1.resize(number_of_top_organisms);
        // std::vector<OrganismLog> top_organisms = std::move(initial_scores_run1);

        // //test top organisms with existing clip
        // input_clip = getExistingMouseInputClip(app);
        // std::vector<OrganismLog> refined_top_scores_run1 = generationRun(cl, snn_engine, app, input_clip, &top_organisms);
        // //debug log
        // Util::Logs::write("NeatPipeline: ran second test on top % ");
        
        // //average top scores
        // std::vector<OrganismLog> refined_scores = averageLogs(std::move(refined_top_scores_run1), std::move(top_organisms));

        // //debug log
        // Util::Logs::write("NeatPipeline: added bottom % to averaged top % ");
        // if(!bottom_organisms.empty()) {
        //     refined_scores.insert(refined_scores.end(), bottom_organisms.begin(), bottom_organisms.end());
        // }
        //sort in descending order
        std::sort(
            refined_scores.begin(), 
            refined_scores.end(), 
            [](const OrganismLog& a, const OrganismLog& b) { return a.raw_fitness > b.raw_fitness; }
        );

        //debug log
        Util::Logs::write("NeatPipeline: sorted refined scores :");
        for(const OrganismLog& log : refined_scores) {
            //debug log
            Util::Logs::write("NeatPipeline: organism " + log.organism_path + " score " + std::to_string(log.raw_fitness));
            printOrganismLog(log);
        }

        std::vector<OrganismLog> adjusted_organisms = Speciation::updateAdjustedFitness(refined_scores, rep_logs);
        //debug log
        Util::Logs::write("NeatPipeline: updated adjusted fitness");
        for(const OrganismLog& log : adjusted_organisms) {
            //debug log
            Util::Logs::write("NeatPipeline: organism " + log.organism_path + " score " + std::to_string(log.adjusted_fitness));
            printOrganismLog(log);
        }
        
        std::map<int, std::vector<OrganismLog>> specie_organisms;
        for (const OrganismLog& log : adjusted_organisms) {
            specie_organisms[log.specie_id].push_back(log);
        }

        //stagnation is decided by generalization on the progress-test clips, before offspring shares
        progressTest(cl, snn_engine, app, specie_organisms);
        //debug log
        Util::Logs::write("NeatPipeline: ran progress test");

        Speciation::updateOffspringShare();
        Speciation::pruneDeadSpecies();
        generation_progress_percentage = 80;
        Util::Logs::write(
            "generation_progress_percentage=" + std::to_string(generation_progress_percentage)
            + "% UNSAFE to shutdown\n");
        //debug log
        Util::Logs::write("NeatPipeline: updated offspring share");

        for (const auto& [specie_id, organisms] : specie_organisms) {
            Reproduction::reproduce(specie_id, organisms);
        }
        generation_progress_percentage = 99;
        Util::Logs::write(
            "generation_progress_percentage=" + std::to_string(generation_progress_percentage)
            + "% UNSAFE to shutdown\n"
        );
        Speciation::saveWorldManager();
        //update world-manager
        //generation ++, reset best_fitness_this_gen,size_this_gen,adjusted_fitness_sum,etc. to 0 for every specie
        Speciation::updateWorldManager();
        Util::JsonAtomicWrite(
            std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/commited-world-manager.json",
            Util::JsonAtomicOpen(std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "species/world-manager.json")
        );
        Util::JsonAtomicWrite(
            std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "app/evolution/commited-innovation-ids.json",
            Util::JsonAtomicOpen(std::filesystem::path(SNN_GPU_PROJECT_ROOT) / "app/evolution/innovation_ids.json")
        );
        generation_progress_percentage = 100;
        Util::Logs::write(
            "generation_progress_percentage=" + std::to_string(generation_progress_percentage)
            + "% UNSAFE to shutdown\n"
        );

        //delete organisms of past generation
        for (const OrganismLog& organism : adjusted_organisms) {
            Speciation::deleteOrganism(std::filesystem::path(organism.organism_path));
        }

        //debug log
        Util::Logs::write("NeatPipeline: reproduced organisms");
    }
}

NeatPipeline::NeatPipeline() = default;
