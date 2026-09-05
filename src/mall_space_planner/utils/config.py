"""YAML configuration loading with inheritance, overrides and path resolution.

Config files may contain a top-level ``_base_`` key (string or list of strings)
pointing to other YAML files that are loaded first and deep-merged. Dotted-key
overrides such as ``train.lr=0.01`` are supported for CLI usage.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary (empty file -> ``{}``)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` and return it."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            # Component specs ({name: ..., params: ...}) are replaced wholesale when the
            # component *name* changes, so base params never leak into another model.
            if "name" in value and "name" in out[key] and value["name"] != out[key]["name"]:
                out[key] = copy.deepcopy(value)
                continue
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: str) -> Any:
    """Best-effort YAML scalar coercion for CLI override strings."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_by_dotted_key(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``dotted_key="a.b"`` (creating dicts as needed)."""
    keys = dotted_key.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def resolve_config(
    path: str | Path,
    overrides: list[str] | dict[str, Any] | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a config with ``_base_`` inheritance, then apply overrides.

    Args:
        path: Path to the YAML file.
        overrides: Either a list of ``"a.b=value"`` strings or a nested dict.
        _seen: Internal cycle guard.

    Returns:
        Fully resolved configuration dictionary. A ``_meta.config_path`` entry
        records the primary file path for provenance.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular _base_ reference detected at {path}")
    _seen.add(path)

    raw = load_yaml(path)
    bases = raw.pop("_base_", None)
    merged: dict[str, Any] = {}
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            b_path = (path.parent / b).resolve() if not Path(b).is_absolute() else Path(b)
            merged = deep_update(merged, resolve_config(b_path, None, _seen))
    merged = deep_update(merged, raw)

    # Machine-local overrides: ``<name>.local.yaml`` next to the config (git-ignored).
    local = path.with_name(f"{path.stem}.local.yaml")
    if local.exists() and local != path:
        merged = deep_update(merged, load_yaml(local))
        merged.setdefault("_meta", {})["local_override"] = str(local)

    if overrides:
        if isinstance(overrides, dict):
            merged = deep_update(merged, overrides)
        else:
            for item in overrides:
                if "=" not in item:
                    raise ValueError(f"Override must look like key=value, got {item!r}")
                k, v = item.split("=", 1)
                set_by_dotted_key(merged, k.strip(), _coerce(v.strip()))

    merged = _expand_env(merged)
    merged.setdefault("_meta", {})["config_path"] = str(path)
    return merged
