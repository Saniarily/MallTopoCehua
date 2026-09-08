"""Outline sources for Stage 3.

An :class:`Outline` is a simplified, metre-scaled polygon (shapely) plus the transform that maps the
source pixel frame to metres, so that real M-node ``CenterPoint`` pixels can be compared with fitted
positions. Three sources:

* ``outline_from_mask``      – the per-floor black/white outline mask PNG (largest white/black blob);
* ``outline_from_total_csv`` – union of all unit polygons in ``{floor}_total.csv`` (shops + corridors +
  ...), buffered/closed and simplified – works without the mask folder;
* ``outline_from_points``    – hand-drawn polygon from the web UI (already in metres or arbitrary units).

Pixel → metre: the export has no explicit scale; if the floor's gross area ``area_m2`` is known (main
table ``total_area`` / number of floors) the scale is ``sqrt(area_m2 / area_px)``, otherwise a default
of ``0.5 m/px`` is used and flagged in ``Outline.scale_source``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

DEFAULT_M_PER_PX = 0.5


@dataclass
class Outline:
    polygon: Polygon  # metres, simplified, valid
    m_per_px: float = 1.0
    origin_px: tuple[float, float] = (0.0, 0.0)  # pixel point mapped to (0, 0) m
    flip_y: bool = True  # image rows grow downwards
    scale_source: str = "given"
    source: str = ""
    extra: dict = field(default_factory=dict)
    # graph-CSV pixel frame -> outline (mask) pixel frame: x' = sx·x + tx, y' = sy·y + ty. The ``*_total.csv`` coordinates
    # and the processed mask / plan PNGs are not always the same pixel frame (crop / padding); see ``align_to_csv``.
    csv_to_px: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)

    # ---- transforms ---------------------------------------------------------------------------
    def csv_px_to_m(self, xy: np.ndarray) -> np.ndarray:
        """Graph-CSV pixels (``*_total.csv`` CenterPoint / coordinates) -> metres in the outline frame."""
        xy = np.asarray(xy, float).reshape(-1, 2).copy()
        sx, sy, tx, ty = self.csv_to_px
        xy[:, 0] = sx * xy[:, 0] + tx
        xy[:, 1] = sy * xy[:, 1] + ty
        return self.px_to_m(xy)

    def px_to_m(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = (xy - np.asarray(self.origin_px)) * self.m_per_px
        if self.flip_y:
            out[:, 1] = -out[:, 1]
        return out

    def m_to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2).copy()
        if self.flip_y:
            xy[:, 1] = -xy[:, 1]
        return xy / self.m_per_px + np.asarray(self.origin_px)

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    def exterior_xy(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.polygon.exterior.coords[:-1]]

    def to_site_boundary(self):  # noqa: ANN201
        from mall_space_planner.schemas import SiteBoundary

        holes = [[(float(x), float(y)) for x, y in r.coords[:-1]] for r in self.polygon.interiors]
        return SiteBoundary(exterior=self.exterior_xy(), holes=holes)


# ------------------------------------------------------------------------------------------ helpers
def _largest(geom) -> Polygon:  # noqa: ANN001
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return max(polys, key=lambda g: g.area) if polys else Polygon()


def _finish(poly_px: Polygon, m_per_px: float, scale_source: str, source: str, simplify_px: float, keep_holes: bool) -> Outline:
    poly_px = poly_px.buffer(0)
    if not keep_holes:
        poly_px = Polygon(poly_px.exterior)
    poly_px = poly_px.simplify(simplify_px, preserve_topology=True)
    minx, miny, maxx, maxy = poly_px.bounds
    origin = (minx, maxy)  # top-left in image coords -> (0,0) m, y up
    o = Outline(polygon=Polygon(), m_per_px=m_per_px, origin_px=origin, scale_source=scale_source, source=source)
    ext = o.px_to_m(np.array(poly_px.exterior.coords))
    holes = [o.px_to_m(np.array(r.coords)) for r in poly_px.interiors]
    o.polygon = Polygon(ext, holes=[h for h in holes if len(h) >= 4]).buffer(0)
    o.polygon = _largest(o.polygon)
    return o


def scale_from_area(area_px: float, area_m2: float | None) -> tuple[float, str]:
    if area_m2 and area_m2 > 0 and area_px > 0:
        return float(np.sqrt(area_m2 / area_px)), "area"
    return DEFAULT_M_PER_PX, "default"


# ------------------------------------------------------------------------------------------ sources
def outline_from_mask(path: str | Path, area_m2: float | None = None, simplify_px: float = 2.0, keep_holes: bool = False) -> Outline:
    """Largest connected blob of a black/white mask PNG → Outline. Foreground = the minority colour
    unless the image is mostly foreground; both conventions are handled by picking the blob that does not
    touch all four image borders."""
    try:
        from PIL import Image
        from skimage.measure import find_contours, label as sk_label, regionprops
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Stage 3 needs Pillow and scikit-image: pip install pillow scikit-image (or conda install -c conda-forge scikit-image pillow)") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    img = np.asarray(Image.open(p).convert("L"))
    bw = img > 127
    H, W = bw.shape
    best: tuple[float, Polygon] | None = None
    for fg in (bw, ~bw):
        lab = sk_label(fg, connectivity=1)
        for rp in regionprops(lab):
            if rp.area < 16:
                continue
            minr, minc, maxr, maxc = rp.bbox
            touches = (minr <= 0) + (minc <= 0) + (maxr >= H) + (maxc >= W)
            if touches >= 3:
                continue  # background blob (mask convention: outline does not touch three image borders)
            mask = np.pad(lab == rp.label, 1)
            for c in find_contours(mask.astype(float), 0.5):
                if len(c) < 4:
                    continue
                poly = Polygon(np.c_[c[:, 1] - 1, c[:, 0] - 1]).buffer(0)  # (row, col) -> (x, y)
                poly = _largest(poly)
                if not poly.is_empty and (best is None or poly.area > best[0]):
                    best = (poly.area, poly)
    if best is None:
        raise ValueError(f"no outline blob found in {path}")
    poly = best[1]
    m_per_px, src = scale_from_area(poly.area, area_m2)
    return _finish(poly, m_per_px, src, str(path), simplify_px, keep_holes)


def read_total_csv(path: str | Path) -> pd.DataFrame:
    """``*_total.csv`` → DataFrame with parsed ``center`` (x, y), ``poly`` (list or None), ``kind`` (letter)."""
    df = pd.read_csv(path)
    df["kind"] = df["index"].astype(str).str[0]

    def _lit(v):  # noqa: ANN001, ANN202
        if isinstance(v, str) and v.strip():
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return None
        return None

    df["center"] = df["CenterPoint"].map(_lit)
    df["poly"] = df["coordinates"].map(_lit)
    for c in ("Related1", "Related2", "Related3"):
        if c in df:
            df[c] = df[c].map(lambda v: _lit(v) or [])
    return df


def outline_from_total_csv(path: str | Path, area_m2: float | None = None, simplify_px: float = 2.0, close_px: float = 20.0, keep_holes: bool = False) -> Outline:
    """Union of all unit polygons (shops, facilities …) closed by a morphological buffer → floor outline.

    In ``*_total.csv`` the corridor nodes (L/M) carry no polygon, so the shop union has corridor-shaped
    gaps.  ``close_px`` must exceed half the widest corridor (≈20 px ≈ 10 m at 0.5 m/px) for the closing
    to fill them; with 6 px about half of the real M junctions fall outside the outline."""
    df = read_total_csv(path)
    polys = []
    for p in df["poly"]:
        if p and len(p) >= 3:
            g = Polygon(p).buffer(0)
            if not g.is_empty:
                polys.append(g)
    if not polys:
        raise ValueError(f"no polygons in {path}")
    u = unary_union(polys).buffer(close_px).buffer(-close_px)
    poly = _largest(u)
    m_per_px, src = scale_from_area(poly.area, area_m2)
    return _finish(poly, m_per_px, src, str(path), simplify_px, keep_holes)


def outline_from_points(points: list[tuple[float, float]], units: str = "m", m_per_unit: float = 1.0, simplify: float = 0.5) -> Outline:
    """Hand-drawn polygon (web UI). ``points`` in metres by default (``m_per_unit`` converts otherwise)."""
    P = np.asarray(points, float) * m_per_unit
    poly = Polygon(P).buffer(0)
    poly = _largest(poly).simplify(simplify, preserve_topology=True)
    return Outline(polygon=poly, m_per_px=1.0, origin_px=(0.0, 0.0), flip_y=False, scale_source="given", source=f"points:{units}")


def m_positions_from_total_csv(path: str | Path, outline: Outline) -> dict[str, tuple[float, float]]:
    """Real M-node positions (metres, same frame as ``outline``) from ``*_total.csv`` – ground truth for Stage 3."""
    df = read_total_csv(path)
    out = {}
    for _, r in df[df["kind"] == "M"].iterrows():
        if r["center"]:
            x, y = outline.csv_px_to_m(np.array(r["center"], float))[0]
            out[str(r["index"])] = (float(x), float(y))
    return out


def corridor_polygons_from_total_csv(path: str | Path, outline: Outline) -> list[Polygon]:
    """Real corridor (L) polygons in metres – reference for width statistics / visual comparison."""
    df = read_total_csv(path)
    out = []
    for _, r in df[df["kind"] == "L"].iterrows():
        if r["poly"] and len(r["poly"]) >= 3:
            g = Polygon(outline.csv_px_to_m(np.array(r["poly"], float))).buffer(0)
            if not g.is_empty:
                out.append(g)
    return out


def align_outline_to_csv(outline: Outline, total_csv: str | Path, min_gain: float = 0.02) -> Outline:
    """Estimate the affine map from the graph-CSV pixel frame to the mask pixel frame.

    The processed mask / plan PNGs (``lrf处理``) are sometimes cropped or padded relative to the plan the graph CSVs were
    digitised from, so the real key points appear shifted (typically down / right) against the outline. The shop unit
    polygons of ``*_total.csv`` must fill the floor plate, so we pick – among identity, bbox-centre translation and
    bbox-to-bbox scale + translation – the map that maximises the IoU of the shop union with the outline.
    A candidate replaces identity only if it gains at least ``min_gain`` coverage; the result is recorded in
    ``outline.extra["csv_align"]``."""
    try:
        df = read_total_csv(total_csv)
    except Exception:  # noqa: BLE001
        return outline
    polys = [Polygon(p).buffer(0) for p in df["poly"] if p and len(p) >= 3]
    polys = [g for g in polys if not g.is_empty]
    if not polys:
        return outline
    shops = unary_union(polys)
    # outline back in mask pixels
    ext_px = outline.m_to_px(np.array(outline.polygon.exterior.coords))
    mask_px = Polygon(ext_px).buffer(0)
    if mask_px.is_empty:
        return outline

    def coverage(sx: float, sy: float, tx: float, ty: float) -> float:
        """IoU of the transformed shop union with the mask (IoU – not plain coverage – separates a genuine rescale
        from a translation that merely drops a smaller shape inside a bigger outline)."""
        from shapely.affinity import affine_transform

        g = affine_transform(shops, [sx, 0, 0, sy, tx, ty])
        inter = g.intersection(mask_px).area
        return float(inter / max(g.union(mask_px).area, 1e-9))

    b_s = shops.bounds; b_m = mask_px.bounds
    ws, hs = b_s[2] - b_s[0], b_s[3] - b_s[1]
    wm, hm = b_m[2] - b_m[0], b_m[3] - b_m[1]
    cands = {"identity": (1.0, 1.0, 0.0, 0.0)}
    cands["translate"] = (1.0, 1.0, (b_m[0] + b_m[2]) / 2 - (b_s[0] + b_s[2]) / 2, (b_m[1] + b_m[3]) / 2 - (b_s[1] + b_s[3]) / 2)
    if ws > 1 and hs > 1:
        sx, sy = wm / ws, hm / hs
        if 0.7 <= sx <= 1.4 and 0.7 <= sy <= 1.4:  # same plan, different resolution / padding; anything else is a different region
            cands["scale"] = (sx, sy, b_m[0] - sx * b_s[0], b_m[1] - sy * b_s[1])
            s_iso = float(np.sqrt(sx * sy))
            cands["scale_iso"] = (s_iso, s_iso, (b_m[0] + b_m[2]) / 2 - s_iso * (b_s[0] + b_s[2]) / 2, (b_m[1] + b_m[3]) / 2 - s_iso * (b_s[1] + b_s[3]) / 2)
    scores = {k: coverage(*v) for k, v in cands.items()}
    best = max(scores, key=lambda k: (scores[k], k == "identity"))
    if best != "identity" and scores[best] >= scores["identity"] + min_gain:
        outline.csv_to_px = tuple(float(x) for x in cands[best])  # type: ignore[assignment]
    outline.extra["csv_align"] = {"mode": best if scores[best] >= scores["identity"] + min_gain else "identity", "coverage_identity": round(scores["identity"], 4), "coverage_best": round(scores[best], 4), "shift_px": (round(outline.csv_to_px[2], 1), round(outline.csv_to_px[3], 1)), "scale": (round(outline.csv_to_px[0], 4), round(outline.csv_to_px[1], 4))}
    return outline


__all__ = ["DEFAULT_M_PER_PX", "align_outline_to_csv", "Outline", "corridor_polygons_from_total_csv", "m_positions_from_total_csv", "outline_from_mask", "outline_from_points", "outline_from_total_csv", "read_total_csv", "scale_from_area"]
