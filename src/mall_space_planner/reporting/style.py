"""Thesis figure style: fonts (CJK-safe), palette, sizes, and a `savefig` that writes png+pdf+svg.

Everything is driven by ``configs/thesis/style.yaml`` so labels/colours/sizes can be edited without
touching Python. SVG/PDF output keeps text as text (``svg.fonttype = none``) so figures can be edited
in Illustrator / Inkscape / PowerPoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_STYLE: dict[str, Any] | None = None
_FONT: str | None = None


def load_style(path: str | Path = "configs/thesis/style.yaml") -> dict[str, Any]:
    global _STYLE
    if _STYLE is None:
        with open(path, encoding="utf-8") as f:
            _STYLE = yaml.safe_load(f)
    return _STYLE


def pick_font(candidates: list[str]) -> str | None:
    installed = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            return c
    return None


def apply_style(style: dict[str, Any] | None = None) -> str:
    """Configure matplotlib rcParams; returns the CJK font actually used."""
    global _FONT
    s = style or load_style()
    cjk = pick_font(s["fonts"]["cjk_candidates"])
    latin = pick_font(s["fonts"]["latin_candidates"]) or "DejaVu Sans"
    family = [f for f in [cjk, latin] if f]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": s["fonts"]["size_base"],
        "axes.titlesize": s["fonts"]["size_title"],
        "axes.labelsize": s["fonts"]["size_label"],
        "xtick.labelsize": s["fonts"]["size_tick"],
        "ytick.labelsize": s["fonts"]["size_tick"],
        "legend.fontsize": s["fonts"]["size_legend"],
        "figure.dpi": 110,
        "savefig.dpi": s["figure"]["dpi"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": s["figure"]["spine_width"],
        "axes.grid": True,
        "grid.alpha": s["figure"]["grid_alpha"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "svg.fonttype": "none",   # keep text editable in SVG
        "pdf.fonttype": 42,       # embed TrueType in PDF (editable text)
        "ps.fonttype": 42,
    })
    _FONT = cjk or latin
    return _FONT


def fig(width: str = "double", height_ratio: float | None = None, **kw: Any):
    s = load_style()
    w = s["figure"]["width_double"] if width == "double" else s["figure"]["width_single"]
    h = w * (height_ratio or s["figure"]["height_ratio"])
    return plt.subplots(figsize=(w, h), **kw)


def color_for(name: str, i: int = 0) -> str:
    s = load_style()
    return s["palette"]["method_colors"].get(name) or s["palette"]["main"][i % len(s["palette"]["main"])]


def label(kind: str, key: str) -> str:
    s = load_style()
    return str(s["labels"].get(kind, {}).get(key, key))


def title_for(code: str) -> str:
    return load_style()["figure_titles"].get(code, code)


def savefig(figure, out_dir: str | Path, stem: str, formats: list[str] | None = None) -> list[Path]:
    s = load_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats or s["figure"]["formats"]:
        p = out_dir / f"{stem}.{ext}"
        figure.savefig(p, bbox_inches="tight", pad_inches=s["figure"]["savefig_pad"], transparent=s["figure"]["transparent"])
        paths.append(p)
    plt.close(figure)
    return paths


def parse_pm(cell: str) -> tuple[float, float]:
    """'0.714 ± 0.009' -> (0.714, 0.009); '0.726' -> (0.726, 0.0)."""
    cell = str(cell).strip()
    if "±" in cell:
        a, b = cell.split("±")
        return float(a), float(b)
    try:
        return float(cell), 0.0
    except ValueError:
        return float("nan"), 0.0


def read_md_table(path: str | Path):
    """Parse a GitHub-markdown table into a DataFrame of strings."""
    import pandas as pd

    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    return pd.DataFrame(rows, columns=header)
