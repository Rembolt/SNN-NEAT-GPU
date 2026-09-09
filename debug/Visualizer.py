#!/usr/bin/env python3
r"""
DNA graph viewer for SNN-GPU genomes.

Usage:
    python debug/Visualizer.py [optional/path/to/dna.json]

Requires:
    pip install PySide6

Controls:
    Mouse wheel: zoom
    Left drag: pan, click neuron circles or arrows for parameters
    Left panel: click any file to load it; list updates live as files appear/disappear
    File > Open DNA: load a file from any directory
    View > Show Connected Neurons Only: avoid zooming out over isolated nodes
    View > Neuron Types: show or hide neuron types
    Auto-reload: the current graph reloads automatically when its file changes on disk
    World manager: double-click a species or general value to graph it across generations
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise SystemExit("Missing GUI dependency. Install it with: pip install PySide6") from exc


TYPE_COLORS = {
    "input": QtGui.QColor(70, 170, 255),
    "hidden": QtGui.QColor(210, 210, 210),
    "output": QtGui.QColor(255, 165, 70),
    "motor": QtGui.QColor(255, 120, 120),
}


def color_for_type(neuron_type: str) -> QtGui.QColor:
    """Stable fallback colors keep new neuron types visually distinct."""
    if neuron_type in TYPE_COLORS:
        return TYPE_COLORS[neuron_type]

    hue = sum((index + 1) * ord(char) for index, char in enumerate(neuron_type)) % 360
    color = QtGui.QColor()
    color.setHsv(hue, 150, 235)
    return color


GraphNode = tuple[QtCore.QPointF, dict]
GraphEdge = tuple[QtCore.QPointF, QtCore.QPointF, dict]


def _layout_positions(neurons: list[dict]) -> dict[int, QtCore.QPointF]:
    """Inputs and bias on the left, hidden scattered in the center, outputs on the right by slot row."""
    inputs = sorted(
        (n for n in neurons if str(n.get("neuron_type")) == "input"),
        key=lambda n: (int(n.get("neuron_type_target", 0)), int(n["innovation_id"])),
    )
    biases = sorted(
        (n for n in neurons if str(n.get("neuron_type")) == "bias"),
        key=lambda n: int(n["innovation_id"]),
    )
    left_neurons = inputs + biases
    center_neurons = [
        n for n in neurons if str(n.get("neuron_type")) in ("hidden", "motor")
    ]
    outputs_by_slot: dict[int, list[dict]] = defaultdict(list)
    for neuron in neurons:
        if str(neuron.get("neuron_type")) != "output":
            continue
        outputs_by_slot[int(neuron.get("neuron_type_target", 0))].append(neuron)
    for slot_neurons in outputs_by_slot.values():
        slot_neurons.sort(key=lambda n: int(n["innovation_id"]))

    column_y_spacing = 14.0
    output_row_y_spacing = 14.0
    output_neuron_x_spacing = 14.0

    sorted_slots = sorted(outputs_by_slot)
    left_height = max(len(left_neurons) - 1, 0) * column_y_spacing
    output_height = max(len(sorted_slots) - 1, 0) * output_row_y_spacing
    center_span_y = max(left_height, output_height, 120.0)

    input_x = 0.0
    center_x_min = 220.0
    center_x_max = 520.0
    output_x = 640.0

    positions: dict[int, QtCore.QPointF] = {}

    left_y_offset = left_height * 0.5
    for row, neuron in enumerate(left_neurons):
        neuron_id = int(neuron["innovation_id"])
        positions[neuron_id] = QtCore.QPointF(input_x, row * column_y_spacing - left_y_offset)

    for neuron in center_neurons:
        neuron_id = int(neuron["innovation_id"])
        rng = random.Random(neuron_id)
        x = rng.uniform(center_x_min, center_x_max)
        y = rng.uniform(-center_span_y * 0.5, center_span_y * 0.5)
        positions[neuron_id] = QtCore.QPointF(x, y)

    output_y_offset = output_height * 0.5
    for row, slot in enumerate(sorted_slots):
        slot_neurons = outputs_by_slot[slot]
        row_y = row * output_row_y_spacing - output_y_offset
        for col, neuron in enumerate(slot_neurons):
            neuron_id = int(neuron["innovation_id"])
            x = output_x + col * output_neuron_x_spacing
            positions[neuron_id] = QtCore.QPointF(x, row_y)

    for neuron in neurons:
        neuron_id = int(neuron["innovation_id"])
        if neuron_id in positions:
            continue
        rng = random.Random(neuron_id)
        x = rng.uniform(center_x_min, center_x_max)
        y = rng.uniform(-center_span_y * 0.5, center_span_y * 0.5)
        positions[neuron_id] = QtCore.QPointF(x, y)

    return positions


def build_layered_graph(
    neurons: list[dict],
    connections: list[dict],
    connected_only: bool = True,
    visible_types: set[str] | None = None,
) -> tuple[dict[str, list[GraphNode]], list[GraphEdge], QtCore.QRectF]:
    enabled_connections = [connection for connection in connections if connection.get("enabled", True) is not False]
    if connected_only and enabled_connections:
        connected_ids = {
            int(connection[node_key])
            for connection in enabled_connections
            for node_key in ("source_node_innovation_id", "target_node_innovation_id")
        }
        neurons = [neuron for neuron in neurons if int(neuron["innovation_id"]) in connected_ids]

    if visible_types is not None:
        neurons = [neuron for neuron in neurons if str(neuron.get("neuron_type", "hidden")) in visible_types]

    positions_by_id = _layout_positions(neurons)
    nodes_by_type: dict[str, list[GraphNode]] = defaultdict(list)
    for neuron in neurons:
        neuron_id = int(neuron["innovation_id"])
        point = positions_by_id[neuron_id]
        nodes_by_type[str(neuron.get("neuron_type", "hidden"))].append((point, neuron))

    radius = 5.0
    margin_x = 80.0
    margin_y = 40.0

    edge_points: list[GraphEdge] = []
    for connection in enabled_connections:
        source = positions_by_id.get(int(connection["source_node_innovation_id"]))
        target = positions_by_id.get(int(connection["target_node_innovation_id"]))
        if source is not None and target is not None:
            edge_points.append((source, target, connection))

    if not positions_by_id:
        return nodes_by_type, edge_points, QtCore.QRectF(-100.0, -100.0, 200.0, 200.0)

    min_x = min(point.x() for point in positions_by_id.values()) - margin_x
    max_x = max(point.x() for point in positions_by_id.values()) + margin_x
    min_y = min(point.y() for point in positions_by_id.values()) - margin_y
    max_y = max(point.y() for point in positions_by_id.values()) + margin_y
    bounds = QtCore.QRectF(QtCore.QPointF(min_x, min_y), QtCore.QPointF(max_x, max_y))
    return nodes_by_type, edge_points, bounds.adjusted(-radius, -radius, radius, radius)


class DnaGraphItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        nodes_by_type: dict[str, list[GraphNode]],
        edge_points: list[GraphEdge],
        bounds: QtCore.QRectF,
    ) -> None:
        super().__init__()
        self.nodes_by_type = nodes_by_type
        self.edge_points = edge_points
        self.bounds = bounds
        self.node_radius = 5.0
        self.arrow_length = 13.0
        self.arrow_width = 8.0
        self.selected_neuron_id: int | None = None
        self.selected_connection_id: int | None = None

    def boundingRect(self) -> QtCore.QRectF:
        return self.bounds

    def paint(self, painter: QtGui.QPainter, _option, _widget=None) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for source, target, connection in self.edge_points:
            if int(connection["innovation_id"]) == self.selected_connection_id:
                arrow_color = QtGui.QColor(255, 240, 120, 230)
                arrow_width = 2.5
            else:
                arrow_color = QtGui.QColor(140, 140, 140, 110)
                arrow_width = 1.0

            painter.setPen(QtGui.QPen(arrow_color, arrow_width))
            painter.setBrush(QtGui.QBrush(arrow_color))
            self._draw_arrow(painter, source, target)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for neuron_type, nodes in self.nodes_by_type.items():
            painter.setBrush(QtGui.QBrush(color_for_type(neuron_type)))
            for point, neuron in nodes:
                painter.drawEllipse(point, self.node_radius, self.node_radius)

                if int(neuron["innovation_id"]) == self.selected_neuron_id:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2.0))
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(point, self.node_radius + 3.0, self.node_radius + 3.0)
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(QtGui.QBrush(color_for_type(neuron_type)))

    def _draw_arrow(self, painter: QtGui.QPainter, source: QtCore.QPointF, target: QtCore.QPointF) -> None:
        dx = target.x() - source.x()
        dy = target.y() - source.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length <= self.node_radius:
            return

        ux = dx / length
        uy = dy / length
        end = QtCore.QPointF(
            target.x() - ux * self.node_radius,
            target.y() - uy * self.node_radius,
        )
        start = QtCore.QPointF(
            source.x() + ux * self.node_radius,
            source.y() + uy * self.node_radius,
        )
        base = QtCore.QPointF(
            end.x() - ux * self.arrow_length,
            end.y() - uy * self.arrow_length,
        )
        px = -uy
        py = ux
        half_width = self.arrow_width * 0.5
        left = QtCore.QPointF(base.x() + px * half_width, base.y() + py * half_width)
        right = QtCore.QPointF(base.x() - px * half_width, base.y() - py * half_width)

        painter.drawLine(start, end)
        painter.drawPolygon(QtGui.QPolygonF([end, left, right]))

    def neuron_at(self, scene_pos: QtCore.QPointF) -> dict | None:
        hit_radius = self.node_radius + 4.0
        hit_radius_squared = hit_radius * hit_radius
        for nodes in self.nodes_by_type.values():
            for point, neuron in nodes:
                dx = scene_pos.x() - point.x()
                dy = scene_pos.y() - point.y()
                if dx * dx + dy * dy <= hit_radius_squared:
                    return neuron
        return None

    def connection_at(self, scene_pos: QtCore.QPointF) -> dict | None:
        hit_distance = 8.0
        best_connection = None
        best_distance_squared = hit_distance * hit_distance

        for source, target, connection in self.edge_points:
            distance_squared = self._distance_to_segment_squared(scene_pos, source, target)
            if distance_squared <= best_distance_squared:
                best_distance_squared = distance_squared
                best_connection = connection

        return best_connection

    def select_neuron(self, neuron: dict | None) -> None:
        self.selected_neuron_id = None if neuron is None else int(neuron["innovation_id"])
        self.selected_connection_id = None
        self.update()

    def select_connection(self, connection: dict | None) -> None:
        self.selected_neuron_id = None
        self.selected_connection_id = None if connection is None else int(connection["innovation_id"])
        self.update()

    @staticmethod
    def _distance_to_segment_squared(
        point: QtCore.QPointF,
        source: QtCore.QPointF,
        target: QtCore.QPointF,
    ) -> float:
        dx = target.x() - source.x()
        dy = target.y() - source.y()
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            point_dx = point.x() - source.x()
            point_dy = point.y() - source.y()
            return point_dx * point_dx + point_dy * point_dy

        t = ((point.x() - source.x()) * dx + (point.y() - source.y()) * dy) / length_squared
        t = max(0.0, min(1.0, t))
        closest = QtCore.QPointF(source.x() + t * dx, source.y() + t * dy)
        closest_dx = point.x() - closest.x()
        closest_dy = point.y() - closest.y()
        return closest_dx * closest_dx + closest_dy * closest_dy


class GraphView(QtWidgets.QGraphicsView):
    neuronClicked = QtCore.Signal(dict)
    connectionClicked = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QtGui.QColor(22, 24, 28))

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            for item in self.scene().items(scene_pos):
                if isinstance(item, DnaGraphItem):
                    neuron = item.neuron_at(scene_pos)
                    if neuron is not None:
                        self.neuronClicked.emit(neuron)
                        event.accept()
                        return

                    connection = item.connection_at(scene_pos)
                    if connection is not None:
                        self.connectionClicked.emit(connection)
                        event.accept()
                        return

        super().mousePressEvent(event)


def flatten_params(params: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(flatten_params(value, full_key))
        else:
            rows.append((full_key, str(value)))
    return rows


_WM_TOP_KEYS = [
    "generation", "population_size", "stagnation_limit", "next_organism_id", "next_species_id",
]

_NA = "N/A"

# ordered keys shown on each species panel; missing keys render as N/A rather than 0
_SPECIES_DISPLAY_KEYS: list[str] = [
    "id",
    "rep_path",
    "rep_id",
    "rep_fitness",
    "created_generation",
    "size_this_gen",
    "offspring_count",
    "best_fitness_this_gen",
    "best_fitness_ever",
    "adjusted_fitness_sum",
    "stagnation_generations",
    "score_on_progress_test",
    "top_one_representation_accuracy",
    "specie_differentiation",
    "differentiation_samples",
    "milliseconds_per_tick",
    "mean_spike_density",
    "max_spike_density",
    "inhibitory_excitatory_ratio",
    "snn_neuron_synapse_ratio",
    "mean_last_click_prediction_error",
    "consistency_last_click_prediction_error",
    "mean_last_click_left_hits",
    "mean_last_click_right_hits",
    "mean_click_type_decisiveness",
    "mean_certainty",
    "calibration_error",
    "soft_jitter",
    "medium_jitter",
    "hard_jitter",
    "learning",
    "mean_anticipation",
    "mean_reaction_time",
    "spike_density_error_ratio",
]

# explicit ordered list of species fields to average, (display_label, species_key)
_SPECIES_AVG_FIELDS: list[tuple[str, str]] = [
    ("average_adjusted_fitness_sum",                    "adjusted_fitness_sum"),
    ("average_best_fitness_ever",                       "best_fitness_ever"),
    ("average_best_fitness_this_gen",                   "best_fitness_this_gen"),
    ("average_calibration_error",                       "calibration_error"),
    ("average_consistency_last_click_prediction_error", "consistency_last_click_prediction_error"),
    ("average_created_generation",                      "created_generation"),
    ("average_hard_jitter",                             "hard_jitter"),
    ("average_inhibitory_excitatory_ratio",             "inhibitory_excitatory_ratio"),
    ("average_learning",                                "learning"),
    ("average_max_spike_density",                       "max_spike_density"),
    ("average_mean_anticipation",                       "mean_anticipation"),
    ("average_mean_certainty",                          "mean_certainty"),
    ("average_mean_click_type_decisiveness",            "mean_click_type_decisiveness"),
    ("average_mean_last_click_left_hits",               "mean_last_click_left_hits"),
    ("average_mean_last_click_prediction_error",        "mean_last_click_prediction_error"),
    ("average_mean_last_click_right_hits",              "mean_last_click_right_hits"),
    ("average_mean_reaction_time",                      "mean_reaction_time"),
    ("average_mean_spike_density",                      "mean_spike_density"),
    ("average_medium_jitter",                           "medium_jitter"),
    ("average_milliseconds_per_tick",                   "milliseconds_per_tick"),
    ("average_score_on_progress_test",                  "score_on_progress_test"),
    ("average_snn_neuron_synapse_ratio",                "snn_neuron_synapse_ratio"),
    ("average_soft_jitter",                             "soft_jitter"),
    ("average_specie_differentiation",                  "specie_differentiation"),
    ("average_spike_density_error_ratio",               "spike_density_error_ratio"),
    ("average_stagnation_generations",                  "stagnation_generations"),
    ("average_top_one_representation_accuracy",         "top_one_representation_accuracy"),
]


def detect_file_type(data: dict) -> str:
    if "species" in data and "population_size" in data:
        return "world_manager"
    return "dna"


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _species_value(sp: dict, key: str) -> str:
    if key not in sp:
        return _NA
    return _fmt_value(sp[key])


def _average_species_field(species_list: list[dict], key: str) -> str:
    values = [s[key] for s in species_list if key in s]
    if not values:
        return _NA
    return _fmt_value(sum(values) / len(values))


def _species_rows(sp: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in _SPECIES_DISPLAY_KEYS:
        rows.append((key, _species_value(sp, key)))
        seen.add(key)
    for key, value in sp.items():
        if key not in seen:
            rows.append((key, _fmt_value(value)))
    return rows


_GEN_FILE_RE = re.compile(r"^(\d+)-generation-.+\.json$")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_TOOLTIP = "Double-click to graph across generations"

# (archive_dir, signature) -> snapshots; reused while generation files are unchanged
_SNAPSHOT_CACHE: tuple[tuple, list[tuple[int, dict]]] | None = None


def _as_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _average_species_number(species_list: list[dict], key: str) -> float | None:
    values = [s[key] for s in species_list if key in s]
    numbers = [n for n in (_as_number(v) for v in values) if n is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _max_species_field(species_list: list[dict], key: str) -> float | None:
    numbers = [
        n for n in (_as_number(s[key]) for s in species_list if key in s) if n is not None
    ]
    if not numbers:
        return None
    return max(numbers)


def _fmt_optional_number(value: float | None) -> str:
    return _NA if value is None else _fmt_value(value)


def _computed_number(data: dict, label: str) -> float | None:
    species_list = data.get("species", [])
    if label == "active_species":
        return float(len(species_list))
    if label == "total_organisms":
        return float(sum(s.get("size_this_gen", 0) for s in species_list))
    if label == "total_offspring":
        return float(sum(s.get("offspring_count", 0) for s in species_list))
    if label == "best_best_fitness_ever":
        if not species_list:
            return None
        return float(max(s.get("best_fitness_ever", 0) for s in species_list))
    if label == "best_best_fitness_this_gen":
        if not species_list:
            return None
        return float(max(s.get("best_fitness_this_gen", 0) for s in species_list))
    if label == "best_score_on_progress_test":
        return _max_species_field(species_list, "score_on_progress_test")
    if label == "best_score_on_progress_test_ever":
        return _max_species_field(species_list, "score_on_progress_test")
    return None


def _spec_number(data: dict, spec: dict) -> float | None:
    kind = spec.get("kind")
    if kind == "species":
        specie_id = spec.get("species_id")
        key = spec.get("key")
        for sp in data.get("species", []):
            if sp.get("id") == specie_id and key in sp:
                return _as_number(sp[key])
        return None
    if kind == "average":
        return _average_species_number(data.get("species", []), spec.get("key", ""))
    if kind == "top":
        key = spec.get("key")
        return _as_number(data[key]) if key in data else None
    if kind == "section":
        sub = data.get(spec.get("section"), {})
        if not isinstance(sub, dict):
            return None
        key = spec.get("key")
        return _as_number(sub[key]) if key in sub else None
    if kind == "computed":
        return _computed_number(data, spec.get("key", ""))
    return None


def _generation_archive_dir(wm_path: Path) -> Path:
    parent = wm_path.parent
    if parent.name == "training":
        return parent
    if parent.name == "species":
        sibling = parent.parent / "training"
        if sibling.is_dir():
            return sibling
    return _PROJECT_ROOT / "records" / "training"


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _file_generation(path: Path, data: dict | None = None) -> int | None:
    match = _GEN_FILE_RE.match(path.name)
    if match:
        return int(match.group(1))
    if data is None:
        data = _read_json(path)
    if data is None:
        return None
    gen = data.get("generation")
    return int(gen) if isinstance(gen, (int, float)) else None


def _load_run_snapshots(wm_path: Path) -> list[tuple[int, dict]]:
    global _SNAPSHOT_CACHE
    archive_dir = _generation_archive_dir(wm_path)
    wm_path = wm_path.resolve()

    files: list[Path] = []
    if archive_dir.is_dir():
        files = [p for p in archive_dir.glob("*.json") if _GEN_FILE_RE.match(p.name)]
    if wm_path not in files and wm_path.exists():
        files.append(wm_path)

    signature_parts: list[tuple[str, int, int]] = []
    entries: list[tuple[float, int, Path]] = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        gen = _file_generation(path)
        if gen is None:
            continue
        signature_parts.append((str(path.resolve()), gen, int(st.st_mtime_ns)))
        entries.append((st.st_mtime, gen, path.resolve()))

    signature = (str(archive_dir.resolve()), str(wm_path), tuple(sorted(signature_parts)))
    if _SNAPSHOT_CACHE is not None and _SNAPSHOT_CACHE[0] == signature:
        return _SNAPSHOT_CACHE[1]

    if not entries:
        return []

    entries.sort(key=lambda e: (e[0], e[1], str(e[2])))
    anchor = next((i for i, e in enumerate(entries) if e[2] == wm_path), len(entries) - 1)

    start = anchor
    while start > 0 and entries[start - 1][1] <= entries[start][1]:
        start -= 1
    end = anchor
    while end + 1 < len(entries) and entries[end + 1][1] >= entries[end][1]:
        end += 1

    newest_by_gen: dict[int, tuple[float, Path]] = {}
    for mtime, gen, path in entries[start : end + 1]:
        prev = newest_by_gen.get(gen)
        if prev is None or mtime >= prev[0]:
            newest_by_gen[gen] = (mtime, path)

    snapshots: list[tuple[int, dict]] = []
    for gen, (_mtime, path) in sorted(newest_by_gen.items()):
        data = _read_json(path)
        if data is not None:
            snapshots.append((gen, data))

    _SNAPSHOT_CACHE = (signature, snapshots)
    return snapshots


def _history_points(snapshots: list[tuple[int, dict]], spec: dict) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for gen, data in snapshots:
        value = _spec_number(data, spec)
        if value is not None:
            points.append((gen, value))
    if spec.get("kind") == "computed" and spec.get("key") == "best_score_on_progress_test_ever":
        running: float | None = None
        ever_points: list[tuple[int, float]] = []
        for gen, value in points:
            running = value if running is None else max(running, value)
            ever_points.append((gen, running))
        return ever_points
    return points


def _best_field_ever(
    snapshots: list[tuple[int, dict]],
    key: str,
    up_to_gen: int | None = None,
) -> float | None:
    best: float | None = None
    for gen, data in snapshots:
        if up_to_gen is not None and gen > up_to_gen:
            continue
        value = _max_species_field(data.get("species", []), key)
        if value is None:
            continue
        best = value if best is None else max(best, value)
    return best


def _history_title(spec: dict) -> str:
    kind = spec.get("kind")
    if kind == "species":
        return f"Species {spec.get('species_id', '?')} · {spec.get('key')}"
    if kind == "section":
        return f"{str(spec.get('section', '')).capitalize()} · {spec.get('key')}"
    if kind == "average":
        return f"Summary · average {spec.get('key')}"
    return f"Summary · {spec.get('key', 'value')}"


_TABLE_STYLE = """
    QTableWidget {
        background-color: rgb(28, 32, 40);
        color: rgb(200, 210, 225);
        gridline-color: rgb(40, 48, 60);
        border: none;
    }
    QHeaderView::section {
        background-color: rgb(35, 40, 52);
        color: rgb(100, 180, 255);
        padding: 2px 4px;
        border: none;
    }
    QTableWidget::item:selected { background-color: rgb(50, 90, 140); }
"""

_ROW_BG_A    = QtGui.QColor(28, 32, 40)
_ROW_BG_B    = QtGui.QColor(32, 36, 46)
_COMPUTED_BG = QtGui.QColor(36, 46, 62)
_NA_FG       = QtGui.QColor(120, 130, 145)


def _make_stat_table() -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, 2)
    table.setHorizontalHeaderLabels(["Parameter", "Value"])
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setStyleSheet(_TABLE_STYLE)
    table.verticalHeader().setDefaultSectionSize(20)
    return table


def _fill_stat_table(
    table: QtWidgets.QTableWidget,
    rows: list[tuple[str, str]],
    computed_start: int = -1,
    series_specs: list[dict | None] | None = None,
) -> None:
    table.setRowCount(len(rows))
    for i, (key, value) in enumerate(rows):
        is_computed = computed_start >= 0 and i >= computed_start
        bg = _COMPUTED_BG if is_computed else (_ROW_BG_A if i % 2 == 0 else _ROW_BG_B)
        spec = series_specs[i] if series_specs is not None and i < len(series_specs) else None
        k_item = QtWidgets.QTableWidgetItem(key)
        v_item = QtWidgets.QTableWidgetItem(value)
        k_item.setBackground(bg)
        v_item.setBackground(bg)
        if value == _NA:
            k_item.setForeground(_NA_FG)
            v_item.setForeground(_NA_FG)
        k_item.setData(QtCore.Qt.ItemDataRole.UserRole, spec)
        tip = f"{key}\n{_HISTORY_TOOLTIP}" if spec else key
        k_item.setToolTip(tip)
        v_item.setToolTip(f"{value}\n{_HISTORY_TOOLTIP}" if spec else value)
        table.setItem(i, 0, k_item)
        table.setItem(i, 1, v_item)
    table.resizeColumnToContents(0)


class HistoryPlot(QtWidgets.QWidget):
    def __init__(self, points: list[tuple[int, float]], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.points = points
        self._hover_index: int | None = None
        self._plot_rect = QtCore.QRectF()
        self.setMinimumSize(360, 180)
        self.setMouseTracking(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(22, 24, 28))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._hover_index = self._nearest_index(event.position())
        self.update()

    def leaveEvent(self, _event: QtCore.QEvent) -> None:
        self._hover_index = None
        self.update()

    def _nearest_index(self, pos: QtCore.QPointF) -> int | None:
        if not self.points or not self._plot_rect.contains(pos):
            return None
        best_i = 0
        best_dx = float("inf")
        for i, (gen, value) in enumerate(self.points):
            pt = self._map_point(gen, value)
            dx = abs(pt.x() - pos.x())
            if dx < best_dx:
                best_dx = dx
                best_i = i
        return best_i

    def _bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, x1 = float(min(xs)), float(max(xs))
        y0, y1 = min(ys), max(ys)
        if x1 <= x0:
            x0 -= 0.5
            x1 += 0.5
        if y1 <= y0:
            pad = abs(y0) * 0.05 if y0 != 0 else 0.5
            y0 -= pad
            y1 += pad
        else:
            pad = (y1 - y0) * 0.08
            y0 -= pad
            y1 += pad
        return x0, x1, y0, y1

    def _map_point(self, gen: float, value: float) -> QtCore.QPointF:
        x0, x1, y0, y1 = self._bounds()
        r = self._plot_rect
        nx = (gen - x0) / (x1 - x0)
        ny = (value - y0) / (y1 - y0)
        return QtCore.QPointF(r.left() + nx * r.width(), r.bottom() - ny * r.height())

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(22, 24, 28))

        if not self.points:
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 130, 145)))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No numeric history")
            return

        left, top, right, bottom = 54, 14, 14, 30
        self._plot_rect = QtCore.QRectF(
            left,
            top,
            max(self.width() - left - right, 1),
            max(self.height() - top - bottom, 1),
        )
        r = self._plot_rect
        x0, x1, y0, y1 = self._bounds()

        painter.setPen(QtGui.QPen(QtGui.QColor(40, 48, 60), 1))
        for i in range(5):
            y = r.top() + r.height() * i / 4
            painter.drawLine(QtCore.QPointF(r.left(), y), QtCore.QPointF(r.right(), y))
            value = y1 - (y1 - y0) * i / 4
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 130, 145)))
            painter.drawText(
                QtCore.QRectF(2, y - 8, left - 6, 16),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{value:.4g}",
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 48, 60), 1))

        tick_count = min(6, max(len({p[0] for p in self.points}), 2))
        for i in range(tick_count):
            gen = x0 + (x1 - x0) * i / (tick_count - 1)
            x = r.left() + r.width() * i / (tick_count - 1)
            painter.setPen(QtGui.QPen(QtGui.QColor(40, 48, 60), 1))
            painter.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom()))
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 130, 145)))
            painter.drawText(
                QtCore.QRectF(x - 28, r.bottom() + 2, 56, 16),
                QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                str(int(round(gen))),
            )

        painter.setPen(QtGui.QPen(QtGui.QColor(70, 80, 100), 1))
        painter.drawRect(r)

        mapped = [self._map_point(gen, value) for gen, value in self.points]
        line_color = QtGui.QColor(100, 180, 255)
        painter.setPen(QtGui.QPen(line_color, 2))
        if len(mapped) == 1:
            painter.setBrush(QtGui.QBrush(line_color))
            painter.drawEllipse(mapped[0], 4, 4)
        else:
            painter.drawPolyline(QtGui.QPolygonF(mapped))
            painter.setBrush(QtGui.QBrush(line_color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for pt in mapped:
                painter.drawEllipse(pt, 3, 3)

        if self._hover_index is not None:
            gen, value = self.points[self._hover_index]
            pt = mapped[self._hover_index]
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 240, 120), 1, QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(QtCore.QPointF(pt.x(), r.top()), QtCore.QPointF(pt.x(), r.bottom()))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 240, 120)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(pt, 5, 5)
            label = f"gen {gen}: {_fmt_value(value)}"
            painter.setPen(QtGui.QPen(QtGui.QColor(230, 235, 245)))
            box = QtCore.QRectF(pt.x() + 8, pt.y() - 22, 160, 18)
            if box.right() > r.right():
                box.moveRight(pt.x() - 8)
            painter.drawText(box, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, label)


class HistoryGraphDialog(QtWidgets.QDialog):
    def __init__(
        self,
        title: str,
        points: list[tuple[int, float]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 420)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(22, 24, 28))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        hdr = QtWidgets.QLabel(title)
        hdr.setStyleSheet("color: rgb(100, 180, 255); font-weight: bold; font-size: 12px;")
        root.addWidget(hdr)

        if points:
            vals = [p[1] for p in points]
            stats = (
                f"{len(points)} generations with this value"
                f"  ·  min {_fmt_value(min(vals))}"
                f"  ·  max {_fmt_value(max(vals))}"
            )
        else:
            stats = "No numeric values found in this run"
        sub = QtWidgets.QLabel(stats)
        sub.setStyleSheet("color: rgb(140, 150, 165); font-size: 11px;")
        root.addWidget(sub)

        plot = HistoryPlot(points)
        root.addWidget(plot, 1)

        table = QtWidgets.QTableWidget(len(points), 2)
        table.setHorizontalHeaderLabels(["Generation", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setStyleSheet(_TABLE_STYLE)
        table.verticalHeader().setDefaultSectionSize(20)
        for i, (gen, value) in enumerate(points):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(gen)))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(_fmt_value(value)))
        table.resizeColumnToContents(0)
        table.setMaximumHeight(160)
        root.addWidget(table)


class WorldManagerView(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(22, 24, 28))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self._current_path: Path | None = None
        self._history_dialogs: list[QtWidgets.QDialog] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        _scroll_style = "QScrollArea { background-color: rgb(22, 24, 28); border: none; }"
        _widget_style = "background-color: rgb(22, 24, 28);"

        # top row: summary + engine/speciation/mutation panels
        self._meta_scroll = QtWidgets.QScrollArea()
        self._meta_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._meta_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._meta_scroll.setWidgetResizable(True)
        self._meta_scroll.setStyleSheet(_scroll_style)

        self._meta_widget = QtWidgets.QWidget()
        self._meta_widget.setStyleSheet(_widget_style)
        self._meta_hbox = QtWidgets.QHBoxLayout(self._meta_widget)
        self._meta_hbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self._meta_hbox.setSpacing(8)
        self._meta_scroll.setWidget(self._meta_widget)

        # bottom row: individual species panels
        species_hdr = QtWidgets.QLabel("Species")
        species_hdr.setStyleSheet("color: rgb(100, 180, 255); font-weight: bold; font-size: 13px;")

        self._species_scroll = QtWidgets.QScrollArea()
        self._species_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._species_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._species_scroll.setWidgetResizable(True)
        self._species_scroll.setStyleSheet(_scroll_style)

        self._species_widget = QtWidgets.QWidget()
        self._species_widget.setStyleSheet(_widget_style)
        self._species_hbox = QtWidgets.QHBoxLayout(self._species_widget)
        self._species_hbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self._species_hbox.setSpacing(8)
        self._species_scroll.setWidget(self._species_widget)

        root.addWidget(self._meta_scroll, 1)
        root.addWidget(species_hdr)
        root.addWidget(self._species_scroll, 1)

    def load(self, data: dict, path: Path | None = None) -> None:
        self._current_path = path
        species_list = data.get("species", [])
        n = len(species_list)

        total_organisms       = sum(s.get("size_this_gen", 0) for s in species_list)
        total_offspring       = sum(s.get("offspring_count", 0) for s in species_list)
        best_fitness_ever     = max((s.get("best_fitness_ever", 0) for s in species_list), default=0)
        best_fitness_this_gen = max((s.get("best_fitness_this_gen", 0) for s in species_list), default=0)
        best_progress_test    = _max_species_field(species_list, "score_on_progress_test")
        current_gen = data.get("generation")
        current_gen_n = int(current_gen) if isinstance(current_gen, (int, float)) else None
        best_progress_test_ever = best_progress_test
        if path is not None:
            snapshots = _load_run_snapshots(path)
            archived_ever = _best_field_ever(snapshots, "score_on_progress_test", current_gen_n)
            if archived_ever is not None:
                best_progress_test_ever = archived_ever
            elif best_progress_test is not None:
                best_progress_test_ever = best_progress_test

        # species with the single best-ever and best-this-gen fitness
        best_ever_sp    = max(species_list, key=lambda s: s.get("best_fitness_ever", 0), default=None)
        best_gen_sp     = max(species_list, key=lambda s: s.get("best_fitness_this_gen", 0), default=None)
        best_legend_path = f"species/{best_ever_sp['id']}-legend.json" if best_ever_sp else _NA
        best_rep_path    = best_gen_sp.get("rep_path", _NA) if best_gen_sp else _NA

        top_keys = [k for k in _WM_TOP_KEYS if k in data]
        top_rows: list[tuple[str, str]] = [
            (k, _fmt_value(data[k])) for k in top_keys
        ]
        top_specs: list[dict | None] = [
            {"kind": "top", "key": k} for k in top_keys
        ]

        computed_rows: list[tuple[str, str]] = [
            ("active_species",             str(n)),
            ("total_organisms",            str(total_organisms)),
            ("total_offspring",            str(total_offspring)),
            ("best_best_fitness_ever",     _fmt_value(best_fitness_ever)),
            ("best_best_fitness_this_gen", _fmt_value(best_fitness_this_gen)),
            ("best_score_on_progress_test",      _fmt_optional_number(best_progress_test)),
            ("best_score_on_progress_test_ever", _fmt_optional_number(best_progress_test_ever)),
            ("best_legend_path",           best_legend_path),
            ("best_rep_path",              best_rep_path),
        ]
        computed_specs: list[dict | None] = [
            {"kind": "computed", "key": "active_species"},
            {"kind": "computed", "key": "total_organisms"},
            {"kind": "computed", "key": "total_offspring"},
            {"kind": "computed", "key": "best_best_fitness_ever"},
            {"kind": "computed", "key": "best_best_fitness_this_gen"},
            {"kind": "computed", "key": "best_score_on_progress_test"},
            {"kind": "computed", "key": "best_score_on_progress_test_ever"},
            None,
            None,
        ]
        if n > 0:
            for label, key in _SPECIES_AVG_FIELDS:
                computed_rows.append((label, _average_species_field(species_list, key)))
                computed_specs.append({"kind": "average", "key": key})

        all_rows = top_rows + computed_rows
        all_specs = top_specs + computed_specs

        for hbox in (self._meta_hbox, self._species_hbox):
            while hbox.count():
                item = hbox.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

        _top = QtCore.Qt.AlignmentFlag.AlignTop
        self._meta_hbox.addWidget(
            self._build_panel(
                "Summary", all_rows, computed_start=len(top_rows), series_specs=all_specs
            ),
            0,
            _top,
        )

        for section in ("engine", "speciation", "mutation"):
            sub = data.get(section)
            if sub:
                rows = [(k, _fmt_value(v)) for k, v in sorted(sub.items())]
                specs = [{"kind": "section", "section": section, "key": k} for k, _v in rows]
                self._meta_hbox.addWidget(
                    self._build_panel(section.capitalize(), rows, series_specs=specs), 0, _top
                )
        self._meta_hbox.addStretch(1)

        for sp in species_list:
            rows = _species_rows(sp)
            specie_id = sp.get("id")
            specs = [{"kind": "species", "species_id": specie_id, "key": k} for k, _v in rows]
            self._species_hbox.addWidget(
                self._build_panel(f"Species {specie_id if specie_id is not None else '?'}", rows, series_specs=specs)
            )
        self._species_hbox.addStretch(1)

    def _on_stat_double_click(self, table: QtWidgets.QTableWidget, row: int) -> None:
        item = table.item(row, 0)
        if item is None:
            return
        spec = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(spec, dict) or self._current_path is None:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            snapshots = _load_run_snapshots(self._current_path)
            points = _history_points(snapshots, spec)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        dlg = HistoryGraphDialog(_history_title(spec), points, self)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        alive: list[QtWidgets.QDialog] = []
        for existing in self._history_dialogs:
            try:
                existing.objectName()
                alive.append(existing)
            except RuntimeError:
                pass
        alive.append(dlg)
        self._history_dialogs = alive
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _build_panel(
        self,
        title: str,
        rows: list[tuple[str, str]],
        computed_start: int = -1,
        series_specs: list[dict | None] | None = None,
    ) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background-color: rgb(28, 32, 40);"
            "         border: 1px solid rgb(50, 60, 80);"
            "         border-radius: 4px; }"
            "QLabel { border: none; background: transparent; }"
        )
        vbox = QtWidgets.QVBoxLayout(frame)
        vbox.setContentsMargins(4, 2, 4, 4)
        vbox.setSpacing(1)

        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet("color: rgb(100, 180, 255); font-weight: bold; font-size: 10px; margin: 0px; padding: 0px;")
        lbl.setFixedHeight(14)
        vbox.addWidget(lbl)

        table = _make_stat_table()
        _fill_stat_table(table, rows, computed_start=computed_start, series_specs=series_specs)
        table.cellDoubleClicked.connect(
            lambda row, _col, t=table: self._on_stat_double_click(t, row)
        )

        row_h    = 20
        header_h = table.horizontalHeader().height()
        table.setFixedHeight(len(rows) * row_h + header_h + 6)
        table.setMinimumWidth(380)
        vbox.addWidget(table)

        frame.setFixedWidth(420)
        frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        return frame


class MultiDirBrowserPanel(QtWidgets.QDockWidget):
    fileSelected = QtCore.Signal(Path)

    def __init__(self, root: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Files", parent)
        self._root = root
        self._current_path: Path | None = None
        self._dirs: list[tuple[str, Path]] = [
            ("Species", root / "species"),
            ("DNAs", root / "DNAs"),
            ("Records", root / "records"),
        ]

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self.setWidget(self._tree)
        self.setMinimumWidth(180)

        self._watcher = QtCore.QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._schedule_refresh)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh()

    def _schedule_refresh(self) -> None:
        # coalesce rapid bursts (e.g. 150 DNA files written per generation) into one rebuild
        self._refresh_timer.start()

    @staticmethod
    def _sort_key(p: Path) -> tuple:
        try:
            return (1, int(p.stem), "")
        except ValueError:
            return (0, -p.stat().st_mtime, "")

    def _populate_dir_group(self, parent_item: QtWidgets.QTreeWidgetItem, dir_path: Path, depth: int = 0) -> int:
        if not dir_path.exists():
            return 0
        self._watcher.addPath(str(dir_path))
        try:
            all_entries = list(dir_path.iterdir())
        except PermissionError:
            return 0

        subdirs = sorted([e for e in all_entries if e.is_dir()], key=lambda p: p.name.lower())
        files = sorted(
            [e for e in all_entries if e.is_file() and e.suffix == ".json" and not e.name.endswith(".json.tmp")],
            key=self._sort_key,
        )

        for f in files:
            item = QtWidgets.QTreeWidgetItem(parent_item, [f.stem])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f)
            item.setToolTip(0, str(f))

        if depth < 2:
            for subdir in subdirs:
                sub_item = QtWidgets.QTreeWidgetItem(parent_item, [subdir.name])
                sub_item.setFlags(sub_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                sub_font = QtGui.QFont()
                sub_font.setItalic(True)
                sub_item.setFont(0, sub_font)
                sub_count = self._populate_dir_group(sub_item, subdir, depth + 1)
                if sub_count == 0:
                    parent_item.removeChild(sub_item)
                else:
                    sub_item.setExpanded(False)

        return len(files) + sum(1 for e in all_entries if e.is_dir())

    def _refresh(self) -> None:
        expanded: set[str] = set()
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.isExpanded():
                expanded.add(item.text(0))
        first_run = self._tree.topLevelItemCount() == 0

        old_dirs = self._watcher.directories()
        if old_dirs:
            self._watcher.removePaths(old_dirs)
        self._tree.clear()

        for label, dir_path in self._dirs:
            group = QtWidgets.QTreeWidgetItem(self._tree, [label])
            group.setFlags(group.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            font = QtGui.QFont()
            font.setBold(True)
            group.setFont(0, font)
            self._populate_dir_group(group, dir_path, depth=0)
            group.setExpanded(label in expanded)

        self._highlight_current()

    def _highlight_current(self) -> None:
        if self._current_path is not None:
            self._find_and_select(self._current_path)

    def _find_and_select(self, path: Path) -> bool:
        it = QtWidgets.QTreeWidgetItemIterator(self._tree)
        while it.value():
            item = it.value()
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == path:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return True
            it += 1
        return False

    def set_current(self, path: Path) -> None:
        self._current_path = path
        self._find_and_select(path)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        path: Path | None = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if path:
            self.fileSelected.emit(path)


class ParameterPanel(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Parameters", parent)
        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setWidget(self.table)
        self.setMinimumWidth(330)

    def show_params(self, title: str, params: dict) -> None:
        self.setWindowTitle(title)
        rows = flatten_params(params)
        self.table.setRowCount(len(rows))

        for row, (key, value) in enumerate(rows):
            key_item = QtWidgets.QTableWidgetItem(key)
            value_item = QtWidgets.QTableWidgetItem(value)
            key_item.setToolTip(key)
            value_item.setToolTip(value)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, value_item)

        self.table.resizeColumnToContents(0)
        self.show()
        self.raise_()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dna_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("DNA Visualizer")
        self.resize(1200, 800)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = GraphView()
        self.view.setScene(self.scene)
        self.view.neuronClicked.connect(self.show_neuron_params)
        self.view.connectionClicked.connect(self.show_connection_params)

        self._wm_view = WorldManagerView()
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self.view)      # index 0: DNA graph
        self._stack.addWidget(self._wm_view)  # index 1: world manager
        self.setCentralWidget(self._stack)

        self.graph_item: DnaGraphItem | None = None
        self.global_params: dict = {}
        self.current_dna_path: Path | None = dna_path
        self._current_file_type: str = "dna"
        self.show_connected_only = True
        self.visible_types: set[str] | None = None
        self.type_actions: dict[str, QtGui.QAction] = {}

        self.param_panel = ParameterPanel(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.param_panel)
        self.param_panel.hide()

        root = Path(__file__).resolve().parents[1]
        self.file_browser = MultiDirBrowserPanel(root, self)
        self.file_browser.fileSelected.connect(self.load_graph)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.file_browser)

        self._file_watcher = QtCore.QFileSystemWatcher()
        self._file_watcher.fileChanged.connect(self._on_file_changed)

        self._reload_timer = QtCore.QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(200)
        self._reload_timer.timeout.connect(self._do_reload)

        self._create_menu()
        self._create_toolbar()
        if dna_path is not None:
            self.load_graph(dna_path)

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QtGui.QAction("&Open DNA...", self)
        open_action.triggered.connect(self.open_dna)
        file_menu.addAction(open_action)

        fit_action = QtGui.QAction("&Fit View", self)
        fit_action.triggered.connect(self.fit_graph)
        file_menu.addAction(fit_action)

        file_menu.addSeparator()
        exit_action = QtGui.QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("&View")
        connected_action = QtGui.QAction("Show &Connected Neurons Only", self)
        connected_action.setCheckable(True)
        connected_action.setChecked(True)
        connected_action.toggled.connect(self.set_connected_only)
        view_menu.addAction(connected_action)

        self.type_menu = view_menu.addMenu("Neuron &Types")
        show_all_types_action = QtGui.QAction("&Show All Types", self)
        show_all_types_action.triggered.connect(self.show_all_types)
        self.type_menu.addAction(show_all_types_action)
        self.type_menu.addSeparator()

    def _create_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Debug Actions", self)
        toolbar.setMovable(False)
        spacer = QtWidgets.QWidget(self)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        global_action = QtGui.QAction("Global Parameters", self)
        global_action.triggered.connect(self.show_global_params)
        toolbar.addAction(global_action)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

    def open_dna(self) -> None:
        start_dir = str(self.current_dna_path.parent) if self.current_dna_path else str(self.file_browser._root)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open DNA JSON",
            start_dir,
            "DNA JSON (*.json);;All Files (*)",
        )
        if path:
            self.load_graph(Path(path))

    def set_connected_only(self, enabled: bool) -> None:
        self.show_connected_only = enabled
        if self.current_dna_path is not None:
            self.load_graph(self.current_dna_path)

    def set_type_visible(self, neuron_type: str, visible: bool) -> None:
        if self.visible_types is None:
            self.visible_types = set(self.type_actions)

        if visible:
            self.visible_types.add(neuron_type)
        else:
            self.visible_types.discard(neuron_type)

        if self.current_dna_path is not None:
            self.load_graph(self.current_dna_path)

    def show_all_types(self) -> None:
        self.visible_types = set(self.type_actions)
        if self.current_dna_path is not None:
            self.load_graph(self.current_dna_path)

    def _on_file_changed(self, path: str) -> None:
        changed = Path(path)
        if changed != self.current_dna_path:
            return
        # re-add to watcher — atomic file replacement (write+rename) drops the watch on Windows
        if not self._file_watcher.files():
            self._file_watcher.addPath(str(changed))
        # debounce: collapse rapid bursts (common with atomic rename on Windows) into one reload
        self._reload_timer.start()

    def _do_reload(self) -> None:
        if self.current_dna_path is not None and self.current_dna_path.exists():
            self.load_graph(self.current_dna_path, _silent=True)

    def load_graph(self, path: Path, *, _silent: bool = False) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            if not _silent:
                QtWidgets.QMessageBox.critical(self, "Could not load file", str(exc))
            return

        if detect_file_type(data) == "world_manager":
            self._load_world_manager(path, data, _silent=_silent)
        else:
            self._load_dna(path, data, _silent=_silent)

    def _update_file_watcher(self, path: Path) -> None:
        current_watched = self._file_watcher.files()
        if current_watched:
            self._file_watcher.removePaths(current_watched)
        self._file_watcher.addPath(str(path))

    def _load_dna(self, dna_path: Path, data: dict, *, _silent: bool = False) -> None:
        try:
            global_params = data.get("global_genome", {})
            neurons = data.get("neuron_genomes", [])
            connections = data.get("connection_genomes", [])
            if not neurons:
                raise ValueError(f"{dna_path} does not contain neuron_genomes")
            all_types = sorted({str(neuron.get("neuron_type", "hidden")) for neuron in neurons})
            self._sync_type_actions(all_types)
            nodes_by_type, edge_points, bounds = build_layered_graph(
                neurons,
                connections,
                connected_only=self.show_connected_only,
                visible_types=self.visible_types,
            )
        except Exception as exc:
            if not _silent:
                QtWidgets.QMessageBox.critical(self, "Could not load DNA", str(exc))
            return

        self._update_file_watcher(dna_path)
        self.file_browser.set_current(dna_path)
        self.global_params = global_params
        self.current_dna_path = dna_path
        self._current_file_type = "dna"
        self.scene.clear()
        self.graph_item = DnaGraphItem(nodes_by_type, edge_points, bounds)
        self.scene.addItem(self.graph_item)
        self.scene.setSceneRect(bounds)
        self._stack.setCurrentIndex(0)
        self.fit_graph()

        type_summary = ", ".join(
            f"{neuron_type}: {len(points)}" for neuron_type, points in sorted(nodes_by_type.items())
        )
        shown_neurons = sum(len(points) for points in nodes_by_type.values())
        mode = "connected only" if self.show_connected_only else "all neurons"
        self.statusBar().showMessage(
            f"{dna_path} | {mode} | shown: {shown_neurons:,}/{len(neurons):,} neurons | "
            f"connections: {len(edge_points):,} | {type_summary}"
        )

    def _load_world_manager(self, path: Path, data: dict, *, _silent: bool = False) -> None:
        try:
            self._wm_view.load(data, path)
        except Exception as exc:
            if not _silent:
                QtWidgets.QMessageBox.critical(self, "Could not load world manager", str(exc))
            return

        self._update_file_watcher(path)
        self.file_browser.set_current(path)
        self.current_dna_path = path
        self._current_file_type = "world_manager"
        self._stack.setCurrentIndex(1)
        self.showMaximized()

        gen = data.get("generation", "?")
        species_list = data.get("species", [])
        self.statusBar().showMessage(
            f"{path} | generation {gen} | {len(species_list)} species"
        )

    def show_neuron_params(self, neuron: dict) -> None:
        if self.graph_item is not None:
            self.graph_item.select_neuron(neuron)

        neuron_id = neuron.get("innovation_id", "?")
        neuron_type = neuron.get("neuron_type", "unknown")
        self.param_panel.show_params(f"Neuron {neuron_id} ({neuron_type})", neuron)

    def show_connection_params(self, connection: dict) -> None:
        if self.graph_item is not None:
            self.graph_item.select_connection(connection)

        connection_id = connection.get("innovation_id", "?")
        source = connection.get("source_node_innovation_id", "?")
        target = connection.get("target_node_innovation_id", "?")
        self.param_panel.show_params(f"Connection {connection_id}: {source} -> {target}", connection)

    def show_global_params(self) -> None:
        if self.graph_item is not None:
            self.graph_item.select_neuron(None)

        self.param_panel.show_params("Global Parameters", self.global_params)

    def _sync_type_actions(self, all_types: list[str]) -> None:
        if self.visible_types is None:
            self.visible_types = set(all_types)

        for neuron_type in all_types:
            if neuron_type in self.type_actions:
                action = self.type_actions[neuron_type]
            else:
                action = QtGui.QAction(neuron_type, self)
                action.setCheckable(True)
                action.toggled.connect(
                    lambda checked, selected_type=neuron_type: self.set_type_visible(selected_type, checked)
                )
                self.type_actions[neuron_type] = action
                self.type_menu.addAction(action)

            action.blockSignals(True)
            action.setChecked(neuron_type in self.visible_types)
            action.blockSignals(False)

    def fit_graph(self) -> None:
        if self.graph_item is None:
            return

        self.view.fitInView(
            self.graph_item.boundingRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view.centerOn(self.graph_item.boundingRect().center())


def main() -> int:
    parser = argparse.ArgumentParser(description="Show a layered neuron/connection map for a DNA JSON file.")
    parser.add_argument("dna", nargs="?", type=Path, default=None, help="Path to a DNA JSON file (optional).")
    args = parser.parse_args()

    if args.dna is not None and not args.dna.exists():
        parser.error(f"DNA file does not exist: {args.dna}")

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args.dna)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
