"""Stage-3 image dataset (``商场平面图dataset - lrf处理``): outline masks, corridor masks, colour plans + ``dataset_0.csv``.

Folder layout (see the dataset's readme.md)::

    outer_mask/    {mall}_{floor}_{region}.png   1 channel, 255 inside the outline
    corridor_mask/ {mall}_{floor}_{region}.png   1 channel, 255 on corridors
    clean_img/     {mall}_{floor}_{region}.png   3 channels, colour-block plan (white background)
    dataset_0.csv  per region: shape_coef, area_total (px²), area_corridor (px²), area_func0..10 (px²)

Region id ``num_id``: 0..n = the n-th indoor region of that floor (multi-region floors are split), −1 = the
floor plan exceeds the image (unusable). Graph-side floor id is ``{mall}_{floor}`` (no region); we join on
region 0 by default and expose the others.

Pixel scale: ``area_total`` is in pixels and the main table's ``total_area`` (m², whole mall) can be
divided over the mall's above-ground floors to estimate ``m_per_px = sqrt(area_m2 / area_px)``; when the
main table is unavailable the default 0.5 m/px is used (``Outline.scale_source`` says which).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mall_space_planner.stage3.outline import DEFAULT_M_PER_PX, Outline, align_outline_to_csv, outline_from_mask, outline_from_total_csv

FUNC_COLOURS_RGB = {  # clean_img colour blocks (from readme.md); func index → (r, g, b)
    "corridor": (252, 249, 242),
    0: (230, 230, 230), 1: (204, 227, 232), 2: (229, 218, 232), 3: (250, 236, 205), 4: (245, 206, 220),
    5: (244, 196, 186), 6: (240, 227, 223), 7: (218, 226, 244), 8: (191, 214, 233), 9: (149, 227, 245), 10: (144, 186, 232),
}


@dataclass
class Stage3Paths:
    outline_mask_dir: Path | None = None
    corridor_mask_dir: Path | None = None
    plan_png_dir: Path | None = None
    dataset_csv: Path | None = None
    graph_dir: Path | None = None
    total_suffix: str = "_total.csv"
    m_per_px: float = DEFAULT_M_PER_PX
    outline_close_px: float = 20.0

    @classmethod
    def from_config(cls, cfg: dict) -> Stage3Paths:
        s3 = cfg.get("stage3", {}) or {}
        ds = (cfg.get("dataset", {}) or {}).get("params", {}) or {}

        def _p(k: str) -> Path | None:
            v = s3.get(k)
            return Path(v).expanduser() if v else None

        gd = ds.get("graph_dir")
        return cls(
            outline_mask_dir=_p("outline_mask_dir"), corridor_mask_dir=_p("corridor_mask_dir"), plan_png_dir=_p("plan_png_dir"),
            dataset_csv=_p("dataset_csv"), graph_dir=Path(gd).expanduser() if gd else None,
            total_suffix=s3.get("total_suffix", "_total.csv"), m_per_px=float(s3.get("m_per_px", DEFAULT_M_PER_PX)),
            outline_close_px=float(s3.get("outline_close_px", 20.0)),
        )

    def image_name(self, floor_id: str, region: int = 0) -> str:
        return f"{floor_id}_{region}.png"

    def outline_mask(self, floor_id: str, region: int = 0) -> Path | None:
        p = self.outline_mask_dir / self.image_name(floor_id, region) if self.outline_mask_dir else None
        return p if p and p.exists() else None

    def corridor_mask(self, floor_id: str, region: int = 0) -> Path | None:
        p = self.corridor_mask_dir / self.image_name(floor_id, region) if self.corridor_mask_dir else None
        return p if p and p.exists() else None

    def plan_png(self, floor_id: str, region: int = 0) -> Path | None:
        p = self.plan_png_dir / self.image_name(floor_id, region) if self.plan_png_dir else None
        return p if p and p.exists() else None

    def total_csv(self, floor_id: str) -> Path | None:
        p = self.graph_dir / f"{floor_id}{self.total_suffix}" if self.graph_dir else None
        return p if p and p.exists() else None


@dataclass
class FloorImageRecord:
    floor_id: str
    mall_id: str
    floor: int
    region: int
    area_total_px: float
    area_corridor_px: float
    shape_coef: float
    func_areas_px: dict[int, float] = field(default_factory=dict)

    @property
    def corridor_ratio(self) -> float:
        return float(self.area_corridor_px / self.area_total_px) if self.area_total_px > 0 else float("nan")


class Stage3Dataset:
    """``dataset_0.csv`` accessor: per-floor corridor ratio / pixel area / region list + outline loader."""

    def __init__(self, paths: Stage3Paths) -> None:
        self.paths = paths
        self.df = pd.DataFrame()
        if paths.dataset_csv and paths.dataset_csv.exists():
            self.df = load_dataset_csv(paths.dataset_csv)

    # ---- records ---------------------------------------------------------------------------------
    def regions(self, floor_id: str) -> list[FloorImageRecord]:
        if self.df.empty:
            return []
        sub = self.df[(self.df["floor_id"] == floor_id) & (self.df["num_id"] >= 0)]
        return [_record(r) for _, r in sub.sort_values("num_id").iterrows()]

    def record(self, floor_id: str, region: int = 0) -> FloorImageRecord | None:
        recs = [r for r in self.regions(floor_id) if r.region == region]
        return recs[0] if recs else None

    def corridor_ratio_stats(self) -> dict[str, float]:
        """Real corridor-area ratio distribution (used to set/justify ``RenderParams.corridor_ratio``)."""
        if self.df.empty:
            return {}
        r = (self.df["area_corridor"] / self.df["area_total"]).replace([np.inf, -np.inf], np.nan).dropna()
        r = r[(r > 0) & (r < 0.8)]
        return {"n": int(len(r)), "mean": float(r.mean()), "median": float(r.median()), "q25": float(r.quantile(0.25)), "q75": float(r.quantile(0.75))}

    # ---- outlines --------------------------------------------------------------------------------
    def m_per_px_for(self, floor_id: str, region: int = 0, mall_area_m2: float | None = None, n_floors: int | None = None) -> tuple[float, str]:
        rec = self.record(floor_id, region)
        if rec and mall_area_m2 and mall_area_m2 > 0 and n_floors and n_floors > 0 and rec.area_total_px > 0:
            return float(np.sqrt((mall_area_m2 / n_floors) / rec.area_total_px)), "main_table_area"
        return self.paths.m_per_px, "default"

    def outline(self, floor_id: str, region: int = 0, area_m2: float | None = None) -> Outline:
        """Outline from the mask PNG when available, else from ``*_total.csv`` polygons. ``area_m2`` = this region's
        gross area in m² (if known) fixes the pixel scale; otherwise the configured default is used."""
        mp = self.paths.outline_mask(floor_id, region)
        if mp is not None:
            o = outline_from_mask(mp, area_m2=area_m2)
            tp = self.paths.total_csv(floor_id)
            if tp is not None:  # the graph CSV pixel frame may be shifted / rescaled against the processed mask PNG
                o = align_outline_to_csv(o, tp)
        else:
            tp = self.paths.total_csv(floor_id)
            if tp is None:
                raise FileNotFoundError(f"no outline source for {floor_id} (mask dir / *_total.csv)")
            o = outline_from_total_csv(tp, area_m2=area_m2, close_px=self.paths.outline_close_px)
        if o.scale_source == "default" and self.paths.m_per_px != DEFAULT_M_PER_PX:
            o = _rescale(o, self.paths.m_per_px)
        rec = self.record(floor_id, region)
        if rec:
            o.extra.update({"real_corridor_ratio": rec.corridor_ratio, "area_total_px": rec.area_total_px, "shape_coef": rec.shape_coef})
        return o

    def has_outline_source(self, floor_id: str, region: int = 0) -> bool:
        return self.paths.outline_mask(floor_id, region) is not None or self.paths.total_csv(floor_id) is not None

    def similar_outlines(self, area_m2: float, k: int = 5, exclude_mall: str | None = None, require_source: bool = True) -> list[str]:
        """Floor ids (region 0) whose pixel area is closest to ``area_m2`` at the default scale – the "pick a real
        outline of similar size" step of the web UI. With ``require_source`` only floors whose mask PNG or
        ``*_total.csv`` actually exists are returned (checked in area order until ``k`` are found)."""
        if self.df.empty:
            return []
        sub = self.df[(self.df["num_id"] == 0) & (self.df["area_total"] > 0)]
        if exclude_mall:
            sub = sub[sub["mall_id"] != exclude_mall]
        target_px = area_m2 / (self.paths.m_per_px**2)
        d = (np.log(sub["area_total"]) - np.log(target_px)).abs()
        ranked = list(sub.assign(d=d).sort_values("d")["floor_id"])
        if not require_source:
            return ranked[:k]
        out: list[str] = []
        for fid in ranked:
            if self.has_outline_source(fid):
                out.append(fid)
                if len(out) >= k:
                    break
        return out


# ------------------------------------------------------------------------------------------ helpers
def load_dataset_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df["mall_id"] = df["mall_id"].astype(str)
    df["floor"] = df["floor"].astype(int)
    df["num_id"] = df["num_id"].astype(int)
    df["floor_id"] = df["mall_id"] + "_" + df["floor"].astype(str)
    return df


def _record(r: pd.Series) -> FloorImageRecord:
    funcs = {int(c[len("area_func"):]): float(r[c]) for c in r.index if str(c).startswith("area_func") and pd.notna(r[c])}
    return FloorImageRecord(
        floor_id=str(r["floor_id"]), mall_id=str(r["mall_id"]), floor=int(r["floor"]), region=int(r["num_id"]),
        area_total_px=float(r["area_total"]) if pd.notna(r["area_total"]) else 0.0,
        area_corridor_px=float(r["area_corridor"]) if pd.notna(r["area_corridor"]) else 0.0,
        shape_coef=float(r["shape_coef"]) if pd.notna(r["shape_coef"]) else float("nan"), func_areas_px=funcs,
    )


def _rescale(o: Outline, m_per_px: float) -> Outline:
    from shapely.affinity import scale as _scale

    f = m_per_px / o.m_per_px
    o.polygon = _scale(o.polygon, xfact=f, yfact=f, origin=(0, 0))
    o.m_per_px = m_per_px
    return o


__all__ = ["FUNC_COLOURS_RGB", "FloorImageRecord", "Stage3Dataset", "Stage3Paths", "load_dataset_csv"]
