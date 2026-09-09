#!/usr/bin/env python3
r"""
Computes pairwise compatibility distances between all living organisms (DNAs/).

Mirrors the C++ Speciation::compareOrganisms / buildComparisonView / compareParams
logic using the same coefficients from Constants.h.

Usage:
    python debug/OrganismDistances.py
    python debug/OrganismDistances.py --dna-dir DNAs
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

# ── speciation constants (mirrors Constants.h :: speciation namespace) ────────
C1_EXCESS        = 1.0   # k_comparison_excess_coeff
C2_DISJOINT      = 1.0   # k_comparison_disjoint_coeff
C3_PARAM         = 1.5   # k_comparison_param_coeff
MIN_GENOME_SIZE  = 20    # k_comparison_min_genome_size

# ── preferred ranges (mirrors Constants.h :: organism_ranges::param_bounds) ───
# compareParams checks this first for normalization; genome_ranges is the fallback
PARAM_BOUNDS_PREFERRED: dict[str, tuple[float, float]] = {
    "leak_reversal_potential":         (-75.0,  -50.0),
    "v_threshold":                     (-55.0,  -40.0),
    "v_rest":                          (-70.0,  -45.0),
    "slope_factor":                    (  1.0,    5.0),
    "a_subthreshold_adaptation":       ( -1.0,    5.0),
    "b_spike_triggered_adaptation":    (  0.0,  100.0),
    "tau_w_adaptation_time_constant":  ( 30.0,  300.0),
    "membrane_capacitance":            (100.0,  300.0),
    "leak_conductance":                ( 10.0,   30.0),
    "spike_peak_voltage":              ( 15.0,   40.0),
    "homeostasis_rate_max_distance":   (  1.5,   10.0),
    "target_spike_rate":               (  0.0,  100.0),
    "initial_weight":                  (-10.0,   10.0),
    "ticks_between_homeostasis":       (  3.0, 2000.0),
    "max_trace_tick":                  ( 10.0,  300.0),
    "max_delay":                       (  1.0,  100.0),
    "feedback_temporal_decay_ticks":   ( 60.0, 1200.0),
    "delay_ticks":                     (  1.0,   90.0),
    "refractory_length_ticks":         (  1.0,   20.0),
}

# ── fallback ranges (mirrors Constants.h :: organism_ranges::genome_ranges) ───
GENOME_RANGES: dict[str, tuple[float, float]] = {
    "membrane_capacitance":            (  1.0, 1000.0),
    "leak_conductance":                (  0.1,  100.0),
    "leak_reversal_potential":         (-90.0,  -40.0),
    "slope_factor":                    (  0.1,   15.0),
    "spike_peak_voltage":              (  0.0,   50.0),
    "homeostasis_weight_mod":          (  0.0,    1.0),
    "homeostasis_pot_bias_mod":        (  0.0,    1.0),
    "homeostasis_dep_bias_mod":        (  0.0,    1.0),
    "homeostasis_threshold_mod":       (  0.0,    1.0),
    "homeostasis_rate_max_distance":   (  0.0,  100.0),
    "ticks_between_homeostasis":       (  1.0, 10000.0),
    "max_trace_tick":                  (  1.0, 1000.0),
    "max_delay":                       (  1.0,  100.0),
    "click_distance_weight":           (  0.0,    1.0),
    "output_axis_decay":               (  0.0,    1.0),
    "feedback_temporal_decay_ticks":   (  1.0, 10000.0),
    "a_subthreshold_adaptation":       (-10.0,   20.0),
    "b_spike_triggered_adaptation":    (  0.0,  400.0),
    "tau_w_adaptation_time_constant":  (  1.0, 1000.0),
    "v_threshold":                     (-70.0,  -30.0),
    "v_rest":                          (-90.0,  -30.0),
    "refractory_length_ticks":         (  1.0,  100.0),
    "target_spike_rate":               (  0.0,  200.0),
    "initial_weight":                  (-100.0, 100.0),
    "delay_ticks":                     (  1.0,  100.0),
    "potentiation_bias":               (  0.0,    1.0),
    "depression_bias":                 (  0.0,    1.0),
    "enabled":                         (  0.0,    1.0),
}

# ── SimpleGene equivalent ─────────────────────────────────────────────────────

def _build_view(organism: dict) -> list[tuple[int, dict[str, float]]]:
    """Build a sorted list of (innovation_id, params) tuples, matching C++ buildComparisonView."""
    gg = organism["global_genome"]
    view: list[tuple[int, dict[str, float]]] = []

    # Global gene uses innovation_id = -1
    view.append((-1, {
        "membrane_capacitance":          float(gg["membrane_capacitance"]),
        "leak_conductance":              float(gg["leak_conductance"]),
        "leak_reversal_potential":       float(gg["leak_reversal_potential"]),
        "slope_factor":                  float(gg["slope_factor"]),
        "spike_peak_voltage":            float(gg["spike_peak_voltage"]),
        "homeostasis_weight_mod":        float(gg["homeostasis_weight_mod"]),
        "homeostasis_pot_bias_mod":      float(gg["homeostasis_pot_bias_mod"]),
        "homeostasis_dep_bias_mod":      float(gg["homeostasis_dep_bias_mod"]),
        "homeostasis_threshold_mod":     float(gg["homeostasis_threshold_mod"]),
        "homeostasis_rate_max_distance": float(gg["homeostasis_rate_max_distance"]),
        "ticks_between_homeostasis":     float(gg.get("ticks_between_homeostasis", 1000)),
        "max_trace_tick":                float(gg["max_trace_tick"]),
        "max_delay":                     float(gg["max_delay"]),
        "click_distance_weight":         float(gg["click_distance_weight"]),
        "output_axis_decay":             float(gg["output_axis_decay"]),
        "feedback_temporal_decay_ticks": float(gg.get("feedback_temporal_decay_ticks", 120)),
    }))

    for neuron in organism["neuron_genomes"]:
        view.append((int(neuron["innovation_id"]), {
            "a_subthreshold_adaptation":    float(neuron["a_subthreshold_adaptation"]),
            "b_spike_triggered_adaptation": float(neuron["b_spike_triggered_adaptation"]),
            "tau_w_adaptation_time_constant": float(neuron["tau_w_adaptation_time_constant"]),
            "v_threshold":                  float(neuron["v_threshold"]),
            "v_rest":                       float(neuron["v_rest"]),
            "refractory_length_ticks":      float(neuron["refractory_length_ticks"]),
            "target_spike_rate":            float(neuron["target_spike_rate"]),
        }))

    for conn in organism["connection_genomes"]:
        enabled_val = 1.0 if conn.get("enabled", False) else 0.0
        view.append((int(conn["innovation_id"]), {
            "initial_weight":    float(conn["initial_weight"]),
            "delay_ticks":       float(conn["delay_ticks"]),
            "potentiation_bias": float(conn["potentiation_bias"]),
            "depression_bias":   float(conn["depression_bias"]),
            "enabled":           enabled_val,
        }))

    view.sort(key=lambda x: x[0])
    return view


def _compare_params(a: dict[str, float], b: dict[str, float]) -> float:
    """Mirrors C++ Speciation::compareParams."""
    if not a:
        return 0.0
    if len(a) != len(b):
        return 1.0
    distance = 0.0
    for key, val_a in a.items():
        if key not in b:
            return 1.0
        # param_bounds preferred range takes priority, genome_ranges is the fallback
        r = PARAM_BOUNDS_PREFERRED.get(key) or GENOME_RANGES.get(key)
        if r is None:
            continue
        rng = abs(r[0] - r[1])
        distance += abs(val_a - b[key]) / rng if rng > 0.0 else 0.0
    return distance / len(a)


def compare_organisms(org_a: dict, org_b: dict) -> float:
    """Mirrors C++ Speciation::compareOrganisms with default speciation coefficients."""
    view_a = _build_view(org_a)
    view_b = _build_view(org_b)

    max_innov_a = view_a[-1][0]
    max_innov_b = view_b[-1][0]

    disjoint = 0
    excess   = 0
    param_diff_sum      = 0.0
    matching_gene_count = 0

    i, j = 0, 0
    while i < len(view_a) and j < len(view_b):
        id_a, params_a = view_a[i]
        id_b, params_b = view_b[j]

        if id_a == id_b:
            param_diff_sum += _compare_params(params_a, params_b)
            matching_gene_count += 1
            i += 1
            j += 1
            continue

        if id_a < id_b:
            if id_a <= max_innov_b:
                disjoint += 1
            else:
                excess += 1
            i += 1
            continue

        if id_b <= max_innov_a:
            disjoint += 1
        else:
            excess += 1
        j += 1

    while i < len(view_a):
        if view_a[i][0] > max_innov_b:
            excess += 1
        else:
            disjoint += 1
        i += 1

    while j < len(view_b):
        if view_b[j][0] > max_innov_a:
            excess += 1
        else:
            disjoint += 1
        j += 1

    genome_size  = max(len(view_a) - 1, len(view_b) - 1)
    normalization = float(genome_size) if genome_size >= MIN_GENOME_SIZE else 1.0

    avg_param_diff = param_diff_sum / matching_gene_count if matching_gene_count > 0 else 0.0

    return (C1_EXCESS * excess + C2_DISJOINT * disjoint) / normalization + C3_PARAM * avg_param_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise organism compatibility distances")
    parser.add_argument("--dna-dir", default="DNAs", help="Directory containing organism JSON files")
    args = parser.parse_args()

    dna_dir = Path(args.dna_dir)
    files = sorted(dna_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {dna_dir}")
        return

    organisms: list[tuple[str, dict]] = []
    for f in files:
        with f.open("r") as fh:
            organisms.append((f.stem, json.load(fh)))

    print(f"Loaded {len(organisms)} organisms from {dna_dir}/")

    if len(organisms) < 2:
        print("Need at least 2 organisms to compare")
        return

    distances: list[float] = []
    min_dist  = math.inf
    max_dist  = -math.inf
    min_pair  = ("", "")
    max_pair  = ("", "")

    pairs = list(combinations(range(len(organisms)), 2))
    total = len(pairs)
    print(f"Computing {total} pairwise distances...")

    for idx, (a, b) in enumerate(pairs):
        if idx % 500 == 0 and idx > 0:
            print(f"  {idx}/{total} ...", flush=True)
        d = compare_organisms(organisms[a][1], organisms[b][1])
        distances.append(d)
        if d < min_dist:
            min_dist = d
            min_pair = (organisms[a][0], organisms[b][0])
        if d > max_dist:
            max_dist = d
            max_pair = (organisms[a][0], organisms[b][0])

    avg_dist = sum(distances) / len(distances)

    print(f"\n── Results ({len(organisms)} organisms, {total} pairs) ──")
    print(f"  Min  : {min_dist:.4f}  ({min_pair[0]} vs {min_pair[1]})")
    print(f"  Max  : {max_dist:.4f}  ({max_pair[0]} vs {max_pair[1]})")
    print(f"  Avg  : {avg_dist:.4f}")


if __name__ == "__main__":
    main()
