"""Reproducibility helpers: seeding and environment fingerprinting."""

from __future__ import annotations

import importlib
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Random seed.
        deterministic_torch: Request deterministic cuDNN kernels when torch is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: WPS433 (optional dependency)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _git_commit(cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _pkg_version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception:  # noqa: BLE001 - any import failure means "absent"
        return None


def collect_environment_info(project_root: str | None = None) -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of the runtime environment.

    Includes Python/platform, git commit, and versions of key packages so that
    checkpoints and experiment reports carry provenance.
    """
    pkgs = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "networkx",
        "shapely",
        "pydantic",
        "torch",
        "torch_geometric",
        "lightgbm",
        "xgboost",
        "catboost",
        "shap",
        "streamlit",
        "mlflow",
    ]
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _git_commit(project_root),
        "packages": {p: _pkg_version(p) for p in pkgs},
    }
    try:
        import torch

        info["torch_device"] = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    except ImportError:
        info["torch_device"] = None
    return info
