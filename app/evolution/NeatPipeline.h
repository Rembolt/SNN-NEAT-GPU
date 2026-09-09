#pragma once
/*
Evolutive proccess
method 1: NEAT loop calls each method
method 2: Transform each initial population DNA into GPU runnable  (DnaReader)
method 3: Run each organism for N ticks (TickLoop(N))
method 4: calculate fitness score (FitnessEvaluation)
method 5: calculate especifiation to place each organism in species and assign species reps (Speciation) 
method 6: calculate adjusted fitness and purge less valuable organisms
method 7: cross-over meiosis with parents of best species (speciation features) (Reproduction)
method 8: Mutation on all new generation (Reproduction)
*/

#include "App.h"
#include "CLBoilerplate.h"
#include "DNAReader.h"
#include "FitnessEvaluation.h"
#include "HostAppParams.h"
#include "HostOrganismBuffers.h"
#include "SNNEngine.h"
#include "DNAviews.h"
#include "Speciation.h"

#include <map>
#include <string>
#include <string_view>

class NeatPipeline {
public:


    NeatPipeline();

    void run();

private:
    void safeStart();
    std::vector<HostMouseInput> getNewMouseInputClip(App& app);
    std::vector<HostMouseInput> getExistingMouseInputClip(App& app);
    std::vector<OrganismLog> loadDnaJsonFiles();
    std::vector<OrganismLog> generationRun(const CLBoilerplate& cl, SNNEngine& snn_engine, App& app, const std::vector<HostMouseInput>& input_clip, const std::vector<OrganismLog>* organism_logs = nullptr);
    void progressTest(const CLBoilerplate& cl, SNNEngine& snn_engine, App& app, const std::map<int, std::vector<OrganismLog>>& specie_organisms);
     
    std::vector<std::filesystem::path> listMouseInputClipFiles(std::string_view subfolder = {});
    void saveMouseInputClip(const std::vector<HostMouseInput>& input_history);
    std::vector<HostMouseInput> loadMouseInputClip(const std::filesystem::path& clip_path,unsigned long expected_tick_count);
    std::vector<HostMouseInput> loadMouseInputClipFromFolder(App& app, unsigned long expected_tick_count);

    int generation_progress_percentage;
};
