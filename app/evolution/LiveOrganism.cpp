#include "LiveOrganism.h"
#include "Constants.h"

#include "FitnessEvaluation.h"
#include "util/Log.h"

namespace {

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

}  // namespace

void LiveOrganism::run()
{
    CLBoilerplate cl;
    cl.printDevice();

    DNAReader dna_reader;
    dna_reader.readDNAFile("species/1288-legend.json");
    HostOrganismBuffers h_organism_buffers = dna_reader.getOrganismBuffers();
    HostAppParams h_app_params = dna_reader.getAppParams();

    SNNEngine snn_engine;
    App app(snn_engine);
    app.setMode(AppMode::Live);

    app.allocateBuffers(
        cl.device(),
        cl.context(),
        cl.queue(),
        h_organism_buffers,
        h_app_params);
    app.initialize();

    FitnessEvaluation fitness_evaluation;
    fitness_evaluation.beginRun(
        live_organism::k_live_tick_count,
        h_organism_buffers.h_global_params_.h_num_neurons,
        h_organism_buffers.h_global_params_.h_num_synapses);

    if (!snn_engine.clearPropagationSpikeRates()) {
        Util::Errors::stop("LiveOrganism: clearPropagationSpikeRates failed");
    }

    Util::Logs::write(
        "LiveOrganism: running Adam for "
        + std::to_string(live_organism::k_live_tick_count)
        + " ticks with ghost overlay.\n");

    app.run(
        live_organism::k_live_tick_count,
        runtime::k_tick_target,
        FitnessScoreCallback,
        &fitness_evaluation);

    std::uint32_t final_inhibitory_spike_rate = 0u;
    std::uint32_t final_excitatory_spike_rate = 0u;
    if (!snn_engine.readPropagationSpikeRates(
            final_inhibitory_spike_rate,
            final_excitatory_spike_rate)) {
        Util::Errors::stop("LiveOrganism: readPropagationSpikeRates failed");
    }

    fitness_evaluation.calculateScoreFactors(
        final_inhibitory_spike_rate,
        final_excitatory_spike_rate);

    const int score = fitness_evaluation.calculateScore();
    Util::Logs::write("LiveOrganism: final score=" + std::to_string(score) + '\n');

    app.freeBuffers();
}
