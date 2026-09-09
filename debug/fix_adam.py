from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DNA_PATH = REPO_ROOT / "DNAs" / "Adam.json"
EVE_DNA_PATH = REPO_ROOT / "DNAs" / "Eve.json"
INNOVATION_IDS_PATH = REPO_ROOT / "app" / "evolution" / "innovation_ids.json"
SEED_SPLIT_CONNECTION_ID = -1

# Must match DNAReader + HostMouseInput / HostFeedbackInput (StructToFloat packing).
VISION_WIDTH = 120
VISION_HEIGHT = 120
VISION_FEATURES = 8
VISION_SLOTS = VISION_WIDTH * VISION_HEIGHT * VISION_FEATURES  # 115200
MOUSE_SLOTS = 7
FEEDBACK_SLOTS = 48
TOTAL_INPUT_SLOTS = MOUSE_SLOTS + FEEDBACK_SLOTS

MOUSE_OUTPUT_X_BINS = 48
MOUSE_OUTPUT_Y_BINS = 48
MOUSE_OUTPUT_CLICK_SLOTS = 2
OUTPUT_SLOT_COUNT = MOUSE_OUTPUT_X_BINS + MOUSE_OUTPUT_Y_BINS + MOUSE_OUTPUT_CLICK_SLOTS
OUTPUT_SLOTS = list(range(OUTPUT_SLOT_COUNT))
# Population per bin; MouseOutput divides slot spikes by this count
OUTPUT_NEURONS_PER_SLOT = 4

OUTPUT_BIAS_WEIGHT = 1.0
HIDDEN_BIAS_WEIGHT = 25.0
DEFAULT_DELAY_TICKS = 1
DEFAULT_POTENTIATION_BIAS = 0.01
DEFAULT_DEPRESSION_BIAS = 0.008

INPUT_TO_HIDDEN_WEIGHT = 10.0
INPUT_TO_HIDDEN_DELAYS = [1, 2, 3]


def max_neuron_innovation_id(neurons: list[dict]) -> int:
    return max(int(n["innovation_id"]) for n in neurons)


def find_neuron(neurons: list[dict], innovation_id: int) -> dict | None:
    for neuron in neurons:
        if neuron["innovation_id"] == innovation_id:
            return neuron
    return None


def next_connection_id(connections: list[dict]) -> int:
    if not connections:
        return 0
    return max(int(connection["innovation_id"]) for connection in connections) + 1


def ensure_split_connection_ids(neurons: list[dict]) -> None:
    for neuron in neurons:
        neuron.setdefault("split_connection_id", SEED_SPLIT_CONNECTION_ID)


def strip_layer_depth(neurons: list[dict]) -> None:
    for neuron in neurons:
        neuron.pop("layer_depth", None)


def normalize_neuron_fields(neuron: dict) -> dict:
    ensure_split_connection_ids([neuron])
    ordered: dict = {
        "innovation_id": neuron["innovation_id"],
        "split_connection_id": neuron["split_connection_id"],
    }
    for key, value in neuron.items():
        if key in ordered or key == "layer_depth":
            continue
        ordered[key] = value
    return ordered


def make_connection(
    innovation_id: int,
    source_id: int,
    target_id: int,
    weight: float,
    *,
    delay_ticks: int | None = None,
) -> dict:
    return {
        "innovation_id": innovation_id,
        "source_node_innovation_id": source_id,
        "target_node_innovation_id": target_id,
        "enabled": True,
        "initial_weight": weight,
        "delay_ticks": DEFAULT_DELAY_TICKS if delay_ticks is None else delay_ticks,
        "potentiation_bias": DEFAULT_POTENTIATION_BIAS,
        "depression_bias": DEFAULT_DEPRESSION_BIAS,
    }


def strip_vision_inputs_and_remap(neurons: list[dict], connections: list[dict]) -> None:
    """Remove input neurons tied to the old screen grid; shift mouse/feedback targets down by VISION_SLOTS."""
    input_nodes = [n for n in neurons if n["neuron_type"] == "input"]
    if input_nodes:
        max_target = max(int(n["neuron_type_target"]) for n in input_nodes)
        # Already remapped to mouse+feedback indices only — do not strip as vision.
        if max_target < TOTAL_INPUT_SLOTS:
            return

    removed_ids: set[int] = set()
    kept: list[dict] = []

    for neuron in neurons:
        if neuron["neuron_type"] != "input":
            kept.append(neuron)
            continue

        target = int(neuron["neuron_type_target"])
        if target < VISION_SLOTS:
            removed_ids.add(int(neuron["innovation_id"]))
            continue

        if target >= VISION_SLOTS + TOTAL_INPUT_SLOTS:
            raise RuntimeError(
                f"Input neuron {neuron['innovation_id']} has out-of-range target {target} "
                f"(expected [{VISION_SLOTS}, {VISION_SLOTS + TOTAL_INPUT_SLOTS - 1}] before remap)"
            )

        neuron["neuron_type_target"] = target - VISION_SLOTS
        kept.append(neuron)

    neurons[:] = kept

    pruned: list[dict] = []
    for connection in connections:
        src = int(connection["source_node_innovation_id"])
        tgt = int(connection["target_node_innovation_id"])
        if src in removed_ids or tgt in removed_ids:
            continue
        pruned.append(connection)
    connections[:] = pruned

    for neuron in neurons:
        if neuron["neuron_type"] != "input":
            continue
        t = int(neuron["neuron_type_target"])
        if not (0 <= t < TOTAL_INPUT_SLOTS):
            raise RuntimeError(
                f"After remap, input neuron {neuron['innovation_id']} has invalid target {t} "
                f"(need [0, {TOTAL_INPUT_SLOTS - 1}])"
            )


def ensure_all_input_neurons(neurons: list[dict]) -> None:
    template = next((n for n in neurons if n["neuron_type"] == "input"), None)
    if template is None:
        raise RuntimeError("Adam.json needs at least one input neuron as template.")

    by_target = {int(n["neuron_type_target"]): n for n in neurons if n["neuron_type"] == "input"}
    next_id = max_neuron_innovation_id(neurons) + 1
    for t in range(TOTAL_INPUT_SLOTS):
        if t not in by_target:
            neuron = deepcopy(template)
            neuron["innovation_id"] = next_id
            neuron["split_connection_id"] = SEED_SPLIT_CONNECTION_ID
            neuron["neuron_type_target"] = t
            neurons.append(neuron)
            next_id += 1


def enabled_connections(connections: list[dict]) -> list[dict]:
    return [connection for connection in connections if connection.get("enabled", True) is not False]


def input_neuron_ids(neurons: list[dict]) -> set[int]:
    return {int(neuron["innovation_id"]) for neuron in neurons if neuron["neuron_type"] == "input"}


def hidden_depth_from_inputs(neurons: list[dict], connections: list[dict]) -> dict[int, int]:
    """Minimum hop distance from any input neuron to each hidden neuron."""
    from collections import deque

    inputs = input_neuron_ids(neurons)
    hidden_ids = {int(neuron["innovation_id"]) for neuron in neurons if neuron["neuron_type"] == "hidden"}
    out_edges: dict[int, list[int]] = {}
    for connection in enabled_connections(connections):
        src = int(connection["source_node_innovation_id"])
        tgt = int(connection["target_node_innovation_id"])
        out_edges.setdefault(src, []).append(tgt)

    depth: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque((input_id, 0) for input_id in inputs)
    while queue:
        node_id, node_depth = queue.popleft()
        for target_id in out_edges.get(node_id, []):
            if target_id not in hidden_ids:
                continue
            next_depth = node_depth + 1
            if target_id in depth and depth[target_id] <= next_depth:
                continue
            depth[target_id] = next_depth
            queue.append((target_id, next_depth))
    return depth


def first_hidden_neurons(neurons: list[dict], connections: list[dict]) -> list[dict]:
    hidden = [neuron for neuron in neurons if neuron["neuron_type"] == "hidden"]
    if not hidden:
        raise RuntimeError("DNA needs at least one hidden neuron.")

    depths = hidden_depth_from_inputs(neurons, connections)
    if depths:
        min_depth = min(depths.values())
        first_layer = [
            neuron
            for neuron in hidden
            if depths.get(int(neuron["innovation_id"])) == min_depth
        ]
    else:
        first_layer = sorted(hidden, key=lambda neuron: int(neuron["innovation_id"]))[:3]

    if not first_layer:
        raise RuntimeError("Could not identify first hidden layer from the connection graph.")
    return sorted(first_layer, key=lambda neuron: int(neuron["innovation_id"]))


def ensure_output_neurons(neurons: list[dict]) -> None:
    template = next((n for n in neurons if n["neuron_type"] == "output"), None)
    if template is None:
        raise RuntimeError("DNA does not contain an output neuron to use as a template.")

    by_slot: dict[int, list[dict]] = {}
    for neuron in neurons:
        if neuron["neuron_type"] != "output":
            continue
        slot = int(neuron["neuron_type_target"])
        by_slot.setdefault(slot, []).append(neuron)

    next_id = max_neuron_innovation_id(neurons) + 1
    for slot in OUTPUT_SLOTS:
        slot_neurons = by_slot.setdefault(slot, [])
        for neuron in slot_neurons:
            neuron["neuron_type"] = "output"
            neuron["neuron_type_target"] = slot
        while len(slot_neurons) < OUTPUT_NEURONS_PER_SLOT:
            output_neuron = deepcopy(template)
            output_neuron["innovation_id"] = next_id
            output_neuron["split_connection_id"] = SEED_SPLIT_CONNECTION_ID
            output_neuron["neuron_type"] = "output"
            output_neuron["neuron_type_target"] = slot
            neurons.append(output_neuron)
            slot_neurons.append(output_neuron)
            next_id += 1


def ensure_input_to_first_hidden(neurons: list[dict], connections: list[dict]) -> None:
    inputs = [n for n in neurons if n["neuron_type"] == "input"]
    hidden1 = first_hidden_neurons(neurons, connections)

    existing_pairs = {
        (int(c["source_node_innovation_id"]), int(c["target_node_innovation_id"])) for c in connections
    }
    next_id = next_connection_id(connections)

    for inp in inputs:
        src = int(inp["innovation_id"])
        for hi, hid in enumerate(hidden1):
            tgt = int(hid["innovation_id"])
            if (src, tgt) in existing_pairs:
                continue
            delay = INPUT_TO_HIDDEN_DELAYS[hi % len(INPUT_TO_HIDDEN_DELAYS)]
            connections.append(
                make_connection(next_id, src, tgt, INPUT_TO_HIDDEN_WEIGHT, delay_ticks=delay)
            )
            existing_pairs.add((src, tgt))
            next_id += 1


def ensure_bias_connections(neurons: list[dict], connections: list[dict]) -> None:
    biases = [n for n in neurons if n["neuron_type"] == "bias"]
    if len(biases) != 1:
        raise RuntimeError(f"Expected exactly one bias neuron, found {len(biases)}")
    bias_id = int(biases[0]["innovation_id"])

    first_hidden_ids = {int(neuron["innovation_id"]) for neuron in first_hidden_neurons(neurons, connections)}
    hidden_bias_targets = sorted(first_hidden_ids)
    outputs_sorted = sorted(
        (n for n in neurons if n["neuron_type"] == "output"),
        key=lambda x: int(x["neuron_type_target"]),
    )
    wanted_targets = hidden_bias_targets + [int(o["innovation_id"]) for o in outputs_sorted]

    blocked_targets = {
        int(n["innovation_id"])
        for n in neurons
        if n["neuron_type"] == "hidden" and int(n["innovation_id"]) not in first_hidden_ids
    }

    connections[:] = [
        c
        for c in connections
        if not (
            int(c["source_node_innovation_id"]) == bias_id
            and int(c["target_node_innovation_id"]) not in wanted_targets
        )
    ]

    seen_bias_targets = {
        int(c["target_node_innovation_id"])
        for c in connections
        if int(c["source_node_innovation_id"]) == bias_id
    }

    next_id = next_connection_id(connections)
    for target_id in wanted_targets:
        if target_id in seen_bias_targets:
            continue
        weight = HIDDEN_BIAS_WEIGHT if target_id in hidden_bias_targets else OUTPUT_BIAS_WEIGHT
        connections.append(make_connection(next_id, bias_id, target_id, weight))
        next_id += 1
        seen_bias_targets.add(target_id)

    for connection in connections:
        if int(connection["source_node_innovation_id"]) != bias_id:
            continue
        target_id = int(connection["target_node_innovation_id"])
        if target_id in hidden_bias_targets:
            connection["initial_weight"] = HIDDEN_BIAS_WEIGHT
        elif target_id in {int(o["innovation_id"]) for o in outputs_sorted}:
            connection["initial_weight"] = OUTPUT_BIAS_WEIGHT
        if target_id in blocked_targets:
            raise RuntimeError(f"Bias still points to blocked target {target_id}")


def dedupe_connections(connections: list[dict]) -> None:
    """Keep one synapse per directed edge (lowest innovation_id wins)."""
    best: dict[tuple[int, int], dict] = {}
    order: list[tuple[int, int]] = []
    for connection in connections:
        key = (
            int(connection["source_node_innovation_id"]),
            int(connection["target_node_innovation_id"]),
        )
        if key not in best:
            best[key] = connection
            order.append(key)
            continue
        if int(connection["innovation_id"]) < int(best[key]["innovation_id"]):
            best[key] = connection
    connections[:] = [best[key] for key in order]


def reset_innovation_ids(neurons: list[dict], connections: list[dict]) -> tuple[int, int]:
    """Neurons 0..N-1 (inputs by target, bias, hidden by graph depth, outputs by slot); connections N..N+M-1."""
    depths = hidden_depth_from_inputs(neurons, connections)
    inputs = sorted(
        (n for n in neurons if n["neuron_type"] == "input"),
        key=lambda x: int(x["neuron_type_target"]),
    )
    biases = [n for n in neurons if n["neuron_type"] == "bias"]
    hidden = sorted(
        (n for n in neurons if n["neuron_type"] == "hidden"),
        key=lambda x: (depths.get(int(x["innovation_id"]), 999), int(x["innovation_id"])),
    )
    outputs = sorted(
        (n for n in neurons if n["neuron_type"] == "output"),
        key=lambda x: int(x["neuron_type_target"]),
    )

    ordered = inputs + biases + hidden + outputs
    if len(ordered) != len(neurons):
        raise RuntimeError(
            "Neuron list contains nodes that are not input/bias/hidden/output "
            f"({len(ordered)} classified vs {len(neurons)} total)"
        )

    old_to_new = {int(old["innovation_id"]): idx for idx, old in enumerate(ordered)}
    for new_id, neuron in enumerate(ordered):
        neuron["innovation_id"] = new_id

    neurons[:] = ordered

    for connection in connections:
        connection["source_node_innovation_id"] = old_to_new[int(connection["source_node_innovation_id"])]
        connection["target_node_innovation_id"] = old_to_new[int(connection["target_node_innovation_id"])]

    n_neurons = len(neurons)
    conn_old_to_new: dict[int, int] = {}
    for i, connection in enumerate(connections):
        old_id = int(connection["innovation_id"])
        new_id = n_neurons + i
        conn_old_to_new[old_id] = new_id
        connection["innovation_id"] = new_id

    for neuron in neurons:
        split_id = int(neuron.get("split_connection_id", SEED_SPLIT_CONNECTION_ID))
        if split_id >= 0:
            neuron["split_connection_id"] = conn_old_to_new[split_id]

    return n_neurons, len(connections)


def write_dna(path: Path, dna: dict) -> None:
    dna["neuron_genomes"] = [normalize_neuron_fields(neuron) for neuron in dna["neuron_genomes"]]
    with path.open("w", encoding="utf-8") as dna_file:
        json.dump(dna, dna_file, indent=2)
        dna_file.write("\n")


def build_innovation_registry(dna_paths: list[Path]) -> dict:
    """Rebuild innovation registry with generation-scoped keys.

    neurons:
      mutated: "split_connection_id,generation" -> neuron_id
      seed:    "neuron_id,-1" -> neuron_id
        (all seeds share split_connection_id=-1, so neuron_id is used in the key)
    connections:
      "source,target,generation" -> connection_id
    Seed genes use generation -1
    """
    connections: dict[str, int] = {}
    neurons: dict[str, int] = {}
    max_id = -1

    for path in dna_paths:
        with path.open("r", encoding="utf-8") as dna_file:
            dna = json.load(dna_file)

        # seed DNA uses generation -1 in registry keys; mutated DNA uses organism generation
        organism_generation = int(dna.get("global_genome", {}).get("generation", 0))
        registry_generation = -1 if organism_generation <= 0 else organism_generation

        for neuron in dna["neuron_genomes"]:
            neuron_id = int(neuron["innovation_id"])
            max_id = max(max_id, neuron_id)
            split_id = int(neuron.get("split_connection_id", SEED_SPLIT_CONNECTION_ID))
            if split_id == SEED_SPLIT_CONNECTION_ID:
                # seeds collide on "-1,-1"; key by neuron_id with generation -1
                neuron_key = f"{neuron_id},-1"
            else:
                neuron_key = f"{split_id},{registry_generation}"
            if neuron_key in neurons and neurons[neuron_key] != neuron_id:
                raise RuntimeError(
                    f"{path.name}: neuron key {neuron_key} has conflicting values "
                    f"({neurons[neuron_key]} vs {neuron_id})"
                )
            neurons[neuron_key] = neuron_id

        for connection in dna["connection_genomes"]:
            conn_id = int(connection["innovation_id"])
            max_id = max(max_id, conn_id)
            src = int(connection["source_node_innovation_id"])
            tgt = int(connection["target_node_innovation_id"])
            edge_key = f"{src},{tgt},{registry_generation}"
            if edge_key in connections:
                if connections[edge_key] != conn_id:
                    connections[edge_key] = min(connections[edge_key], conn_id)
            else:
                connections[edge_key] = conn_id

    def neuron_sort_key(item: tuple[str, int]) -> tuple[int, int]:
        left, right = item[0].split(",", 1)
        return (int(left), int(right))

    def connection_sort_key(item: tuple[str, int]) -> tuple[int, int, int]:
        src, tgt, gen = item[0].split(",", 2)
        return (int(src), int(tgt), int(gen))

    return {
        "next_id": max_id + 1,
        "neurons": dict(sorted(neurons.items(), key=neuron_sort_key)),
        "connections": dict(sorted(connections.items(), key=connection_sort_key)),
    }


def write_innovation_registry(path: Path, registry: dict) -> None:
    with path.open("w", encoding="utf-8") as registry_file:
        json.dump(registry, registry_file, indent=2)
        registry_file.write("\n")


def fix_dna_topology(dna: dict) -> tuple[int, int]:
    global_genome = dna["global_genome"]
    global_genome.pop("vision_params", None)

    neurons = dna["neuron_genomes"]
    connections = dna["connection_genomes"]

    strip_vision_inputs_and_remap(neurons, connections)
    ensure_all_input_neurons(neurons)
    ensure_output_neurons(neurons)
    ensure_input_to_first_hidden(neurons, connections)
    ensure_bias_connections(neurons, connections)
    dedupe_connections(connections)
    strip_layer_depth(neurons)
    ensure_split_connection_ids(neurons)
    return reset_innovation_ids(neurons, connections)


def main() -> None:
    results: list[tuple[Path, int, int]] = []
    for path in (DNA_PATH, EVE_DNA_PATH):
        with path.open("r", encoding="utf-8") as dna_file:
            dna = json.load(dna_file)
        n_neurons, n_synapses = fix_dna_topology(dna)
        write_dna(path, dna)
        results.append((path, n_neurons, n_synapses))

    registry = build_innovation_registry([DNA_PATH, EVE_DNA_PATH])
    write_innovation_registry(INNOVATION_IDS_PATH, registry)

    for path, n_neurons, n_synapses in results:
        print(f"Updated {path}")
        print(
            f"  neurons: {n_neurons} "
            f"(inputs 0..{TOTAL_INPUT_SLOTS - 1}, "
            f"outputs 0..{OUTPUT_SLOT_COUNT - 1} x{OUTPUT_NEURONS_PER_SLOT}), "
            f"synapses: {n_synapses}"
        )
        print(
            f"  innovation IDs: neurons 0..{n_neurons - 1}, "
            f"connections {n_neurons}..{n_neurons + n_synapses - 1}"
        )
    print(f"Wrote {INNOVATION_IDS_PATH}")
    print(
        f"Registry: next_id={registry['next_id']}, "
        f"{len(registry['neurons'])} neurons, "
        f"{len(registry['connections'])} connections"
    )


if __name__ == "__main__":
    main()
