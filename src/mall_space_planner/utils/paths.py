"""Project path management. No hard-coded absolute paths anywhere else in the package."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root.

    Resolution order: ``MSP_PROJECT_ROOT`` environment variable, otherwise the
    directory containing ``pyproject.toml`` found by walking up from this file.
    Falls back to the current working directory.
    """
    env = os.environ.get("MSP_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass
class ProjectPaths:
    """Canonical directory layout, all relative to :attr:`root` unless overridden."""

    root: Path = field(default_factory=get_project_root)
    configs: Path | None = None
    data_raw: Path | None = None
    data_interim: Path | None = None
    data_processed: Path | None = None
    data_samples: Path | None = None
    outputs: Path | None = None
    checkpoints: Path | None = None
    experiments: Path | None = None
    reports: Path | None = None
    figures: Path | None = None
    generated_layouts: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        defaults = {
            "configs": "configs",
            "data_raw": "data/raw",
            "data_interim": "data/interim",
            "data_processed": "data/processed",
            "data_samples": "data/samples",
            "outputs": "outputs",
            "checkpoints": "outputs/checkpoints",
            "experiments": "outputs/experiments",
            "reports": "outputs/reports",
            "figures": "outputs/figures",
            "generated_layouts": "outputs/generated_layouts",
        }
        for name, rel in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, self.root / rel)
            else:
                setattr(self, name, Path(getattr(self, name)))

    def ensure(self) -> ProjectPaths:
        """Create all managed directories if missing."""
        for name in (
            "data_raw",
            "data_interim",
            "data_processed",
            "data_samples",
            "checkpoints",
            "experiments",
            "reports",
            "figures",
            "generated_layouts",
        ):
            Path(getattr(self, name)).mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, p: str | Path) -> Path:
        """Resolve ``p`` relative to the project root when it is not absolute."""
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p)
