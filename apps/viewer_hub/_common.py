"""Shared bootstrap for the Streamlit Viewer Hub pages.

Only wiring lives here (sys.path, cached resource handles, small widgets). All algorithms are in
``mall_space_planner.hub.*`` and are the same objects the FastAPI backend serves.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mall_space_planner.hub.catalog import Catalog  # noqa: E402
from mall_space_planner.hub.viz import ensure_style  # noqa: E402
from mall_space_planner.hub.experiments import ExperimentRegistry  # noqa: E402
from mall_space_planner.hub.jobs import JobRunner  # noqa: E402
from mall_space_planner.hub.workbench import Workbench  # noqa: E402

CONFIG_DIR = ROOT / "configs"
ensure_style()  # CJK font chain for every matplotlib figure in the hub


def setup_page(title: str, icon: str = "🏬") -> None:
    st.set_page_config(page_title=f"{title} · MallTopoCehua", page_icon=icon, layout="wide")
    st.title(title)


@st.cache_resource(show_spinner="载入案例库 …")
def get_catalog(preferred: str | None = None) -> Catalog:
    return Catalog.load(preferred)


@st.cache_resource(show_spinner="载入 / 拟合阶段一排序器与阶段二生成器 …")
def get_workbench(stage1_config: str = "configs/stage1/extra_trees.yaml", stage2_config: str = "configs/stage2/search_baseline.yaml") -> Workbench:
    return Workbench.load(get_catalog(), stage1_config, stage2_config)


@st.cache_resource
def get_jobs() -> JobRunner:
    return JobRunner()


def get_registry() -> ExperimentRegistry:
    """Not cached on purpose: results appear as soon as a job writes them."""
    return ExperimentRegistry().scan()


def fig_png(fig, dpi: int = 150) -> bytes:  # noqa: ANN001
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show_fig(fig, dpi: int = 130, **kw) -> None:  # noqa: ANN001
    st.image(fig_png(fig, dpi), **kw)


def list_configs(sub: str) -> list[str]:
    d = CONFIG_DIR / sub
    return sorted(str(p.relative_to(ROOT)) for p in d.glob("*.yaml")) if d.exists() else []


def data_banner(cat: Catalog) -> None:
    ov = cat.overview()
    if not ov.get("is_real"):
        st.info(f"当前载入的是 **合成数据集**（{ov['n_floors']} 层 / {ov['n_malls']} 个商场，`{ov['path']}`）。在 Mac 上运行时若 `data/processed/legacy` 存在会自动切换到真实数据。", icon="ℹ️")
    else:
        st.caption(f"数据：{ov['path']} · {ov['n_floors']} 层 / {ov['n_malls']} 个商场")
