# SNN-NEAT-GPU

**A spiking neural network built from scratch in C++ and OpenCL that learns to predict mouse position and clicks, with both its topology and neuron dynamics evolved through NEAT instead of backpropagation**

[Demo GIF — live cursor prediction driven by the evolved SNN]

## Why this project

Most neural-network projects rely on established tensor libraries; this one contains the complete simulation and learning pipeline as sparse OpenCL kernels. The challenge is twofold: preserve spike timing, delayed transmission, and local plasticity efficiently on a GPU while searching a large, discontinuous space of network structures and biologically inspired parameters.

The result is an end-to-end neuroevolution system covering GPU compute, sparse data structures, evolutionary algorithms, real-time input capture, fault-tolerant training state, and custom visualization.

## System architecture

`Mouse input → spike encoding → AdEx neuron update → spike compaction → reward-modulated STDP → delayed propagation → output decoding → fitness → NEAT reproduction`

- **Simulation core:** Adaptive Exponential Integrate-and-Fire neurons, refractory periods, firing-rate traces, and homeostatic regulation
- **Sparse GPU pipeline:** CSR adjacency for outgoing propagation, CSC adjacency for incoming STDP updates, packed dirty bits, and circular buffers for per-synapse delays
- **OpenCL kernels:** input encoding, neuron integration, active-spike compaction, plasticity, propagation, and homeostasis run as separate stages
- **Learning:** reward-modulated spike-timing-dependent plasticity updates weights during each evaluation
- **Evolution:** NEAT-style structural and parameter mutation, innovation tracking, tournament selection, elitism, stagnation handling, and adaptive speciation
- **Behavioral diversity:** species are grouped by sampled cursor-position and click behavior rather than genome distance
- **Evaluation:** every organism replays the same recorded mouse-input clip; separate progress-test clips measure generalization

[Evolved topology — visualizer view of a representative species]

[Spike activity — raster or firing-density plot during inference]

## Results

The latest training run reached **677 generations** with a population of **150** and maintained **6 active species**. The recorded best raw fitness was **5,020**; at generation 671, the best held-out progress-test score was **3,738**, while species logs reported approximately **0.23 ms per simulation tick** on the development machine. Fitness values are internal optimization scores, not external benchmark results.

[Training plot — best and average fitness across generations]

Progress was real but ultimately plateaued. Minimal starting topologies, better-scaled mutations, corrected speciation, and revised fitness signals moved training beyond the early dead ends; however, later generations still converged on incomplete strategies. For example, a leading generation-671 species achieved roughly **95.6% left-click hits but 0% right-click hits**, exposing behavioral collapse rather than a solved predictor.

## Development history

- **Foundation (days 1–3):** Built the initial simulator and NEAT pipeline, then traced early stagnation to over-complex seed genomes, incorrect dynamic-speciation math, and mutation-distribution bugs
- **Evolution stability (days 4–7):** Introduced parameter-specific mutation scales, corrected firing-rate decay, stabilized the species count, raised the population from 70 to 150, and reworked elitism and stagnation safeguards
- **Observability and speed (days 8–12):** Removed scoring caps, expanded generation/species logs, made saves recoverable, reused compiled kernels, and reduced per-organism setup overhead; the faster run still plateaued after 370 generations
- **Search-space redesign (days 13–18):** Started from minimal structured topologies, reduced output encoding size, increased stagnation patience, and replaced competing anticipation/location objectives with one time-aware squared prediction error
- **Generalization and diversity (days 19–23):** Removed uneven retesting, switched from genotypic to behavioral speciation, added fixed progress-test clips, and seeded three complexity levels (Adam, Abel, and Eve). Training improved early, then stalled again by generation 677

## Limitations and future work

- The evolved policy did not become a reliable general mouse predictor; click-side collapse and long fitness plateaus remain
- The next investigation should isolate the effect of neuron-to-synapse density, species stagnation limits, and behavior-distance thresholds
- Fitness still provides delayed, indirect credit for a temporal SNN task; richer novelty or curriculum signals may improve credit assignment
- GPU occupancy and host/device synchronization have not been profiled systematically across hardware
- The current build and input-capture path target Windows; portability work is still needed

## Run locally

### Requirements

- Windows with a working OpenCL SDK/runtime and a GPU or CPU OpenCL device
- CMake 3.25+, a C++20 MinGW toolchain, and GLFW 3
- Internet access during the first configure step to fetch `nlohmann/json`

### Live inference

The live preset loads the checked-in representative genome and displays its predicted position as a ghost cursor:

```powershell
cmake --preset live
cmake --build --preset live
.\build-live\snn_app.exe
```

### Training

Training continuously records/replays mouse-input clips, evaluates the population, updates species, and writes recoverable generation state:

```powershell
cmake --preset train
cmake --build --preset train
.\build-train\snn_app.exe
```

Training expects seed genome JSON files in `DNAs/`. Archived seed genomes and matching world-state files are available under `records/starter-files/`.

### Visualizer

```powershell
python -m pip install PySide6
python debug/Visualizer.py species/1858-legend.json
```

The visualizer supports interactive genome inspection and plots metrics directly from generation records.

## Tech stack

C++20 · OpenCL · CMake · GLFW · nlohmann/json · Python · PySide6

## License

MIT