"""Parser for the Stage-2 ShareGPT corpus (``sharegpt_data.json``).

Each record is a two-turn conversation. The *human* turn is a templated prompt::

    # Context: Commercial Floor Plan Design
    # City: 北京市, Layout: 一字型, Area:about 48400.0 sqm
    # Target_Scale: Approx 20 nodes
    # Task: Expand skeleton_graph into complete_topology.

    # Skeleton Input (Core Structure)
    skeleton_graph = { "A": ["B", "D"], ... }

    # Generated Complete Topology (JSON format)
    complete_topology =

The *gpt* turn is a fenced JSON adjacency list. Nodes are letter labels (``A``..``Z``,
``AA``..). Facts verified on the file (see ``docs/data_audit.md``): 5 632 records, all
skeleton edges preserved in the target (edge recall 1.0), adjacency lists symmetric,
~79 % of targets connected, 397 records with ``Unknown_Layout`` / ``Area:about 0``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from mall_space_planner.schemas import LayoutType, TopologyGraph
from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

_RE_CITY = re.compile(r"City:\s*([^,\n]+),")
_RE_LAYOUT = re.compile(r"Layout:\s*([^,\n]+),")
_RE_AREA = re.compile(r"Area:\s*about\s*([0-9.]+)")
_RE_SCALE = re.compile(r"Approx\s*(\d+)\s*nodes")
_RE_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class ExpansionSample:
    """One skeleton → complete-topology supervision pair."""

    sample_id: str
    city: str | None
    layout_type: LayoutType | None
    area_sqm: float | None
    target_num_nodes: int | None
    skeleton: TopologyGraph
    target: TopologyGraph

    def to_record(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "city": self.city,
            "layout_type": self.layout_type.value if self.layout_type else None,
            "area_sqm": self.area_sqm,
            "target_num_nodes": self.target_num_nodes,
            "skeleton": self.skeleton.adjacency,
            "target": self.target.adjacency,
        }

    @classmethod
    def from_record(cls, rec: dict) -> ExpansionSample:
        lt = rec.get("layout_type")
        return cls(
            sample_id=rec["sample_id"],
            city=rec.get("city"),
            layout_type=LayoutType(lt) if lt else None,
            area_sqm=rec.get("area_sqm"),
            target_num_nodes=rec.get("target_num_nodes"),
            skeleton=TopologyGraph(adjacency=rec["skeleton"]),
            target=TopologyGraph(adjacency=rec["target"]),
        )


def _parse_adjacency(text: str) -> dict[str, list[str]]:
    m = _RE_JSON.search(text)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("adjacency must be a JSON object")
    return {str(k): [str(x) for x in v] for k, v in obj.items()}


def _layout(value: str | None) -> LayoutType | None:
    if value is None:
        return None
    value = value.strip()
    try:
        return LayoutType(value)
    except ValueError:
        logger.debug("Unrecognised layout label %r", value)
        return LayoutType.UNKNOWN


def parse_sharegpt_record(rec: dict, sample_id: str) -> ExpansionSample:
    """Parse a single ShareGPT conversation into an :class:`ExpansionSample`."""
    conv = rec["conversations"]
    human = next(c["value"] for c in conv if c.get("from") == "human")
    gpt = next(c["value"] for c in conv if c.get("from") == "gpt")

    city = (_RE_CITY.search(human) or [None, None])[1]
    layout = (_RE_LAYOUT.search(human) or [None, None])[1]
    area = _RE_AREA.search(human)
    scale = _RE_SCALE.search(human)

    sk_start = human.index("skeleton_graph")
    skeleton = _parse_adjacency(human[sk_start:])
    target = _parse_adjacency(gpt)

    area_val = float(area.group(1)) if area else None
    if area_val is not None and area_val <= 0:
        area_val = None  # "Area:about 0 sqm" means unknown

    return ExpansionSample(
        sample_id=sample_id,
        city=city.strip() if city else None,
        layout_type=_layout(layout),
        area_sqm=area_val,
        target_num_nodes=int(scale.group(1)) if scale else None,
        skeleton=TopologyGraph(adjacency=skeleton),
        target=TopologyGraph(adjacency=target),
    )


def load_sharegpt(path: str | Path, limit: int | None = None) -> list[ExpansionSample]:
    """Load and parse the corpus. Malformed records are skipped with a warning."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[ExpansionSample] = []
    n_bad = 0
    for i, rec in enumerate(data):
        if limit is not None and len(out) >= limit:
            break
        try:
            out.append(parse_sharegpt_record(rec, sample_id=f"sg_{i:05d}"))
        except Exception as exc:  # noqa: BLE001
            n_bad += 1
            logger.warning("Skipping ShareGPT record %d: %s", i, exc)
    logger.info("Loaded %d expansion samples from %s (%d skipped)", len(out), path, n_bad)
    return out
