#include "util/Log.h"

#include <exception>
#include <string>

#if defined(SNN_GPU_APP_TRAIN)
#include "NeatPipeline.h"
#elif defined(SNN_GPU_APP_LIVE)
#include "LiveOrganism.h"
#else
#error "Configure with -DSNN_GPU_APP_MODE=train|train-debug|live|live-debug"
#endif

#if defined(SNN_GPU_RUN_TESTS)
bool runSpeciationFileTests();
#endif

int main()
{
    try {
#if defined(SNN_GPU_RUN_TESTS)
        if (!runSpeciationFileTests()) {
            return 1;
        }
        return 0;
#endif
#if defined(SNN_GPU_APP_TRAIN)
        NeatPipeline neat_pipeline;
        neat_pipeline.run();
#elif defined(SNN_GPU_APP_LIVE)
        LiveOrganism live_organism;
        live_organism.run();
#endif
    } catch (const std::exception& e) {
        Util::Errors::write("Error: " + std::string(e.what()) + '\n');
        return 1;
    }
    return 0;
}
