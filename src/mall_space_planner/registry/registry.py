"""Minimal, dependency-free registry/factory implementation."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from mall_space_planner.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Component kinds recognised by the framework. Registering an unknown kind is allowed
# (for extensions) but a warning is logged.
KNOWN_KINDS: tuple[str, ...] = (
    # stage 1
    "retriever",
    "ranker",
    "graph_encoder",
    "fusion",
    "calibrator",
    "explainer",
    "stage1_model",
    # stage 2
    "generator",
    "geometry_decoder",
    "constraint_solver",
    "repairer",
    "evaluator",
    # shared
    "feature_builder",
    "dataset_adapter",
    "loss",
)

# Modules that contain built-in registrations. They are imported lazily by
# :func:`import_builtin_components` so that importing the registry itself stays cheap
# and optional dependencies (torch, lightgbm, ...) are only touched when needed.
BUILTIN_MODULES: tuple[str, ...] = (
    "mall_space_planner.data.adapters",
    "mall_space_planner.features.builders",
    "mall_space_planner.stage1.retrievers",
    "mall_space_planner.stage1.rankers",
    "mall_space_planner.stage1.explainers",
    "mall_space_planner.stage2.generators",
    "mall_space_planner.stage2.decoders",
    "mall_space_planner.stage2.repair",
    "mall_space_planner.evaluation",
)


class Registry:
    """Two-level mapping ``kind -> name -> class`` with a config-driven factory."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, type]] = {}
        self._builtins_loaded = False

    # ------------------------------------------------------------------ registration
    def register(self, kind: str, name: str, *, override: bool = False) -> Callable[[type[T]], type[T]]:
        if kind not in KNOWN_KINDS:
            logger.warning("Registering component under non-standard kind %r", kind)

        def deco(cls: type[T]) -> type[T]:
            bucket = self._store.setdefault(kind, {})
            if name in bucket and not override and bucket[name] is not cls:
                raise KeyError(f"{kind}/{name} already registered as {bucket[name]!r}")
            bucket[name] = cls
            setattr(cls, "registry_kind", kind)
            setattr(cls, "registry_name", name)
            return cls

        return deco

    # ------------------------------------------------------------------ lookup
    def get(self, kind: str, name: str) -> type:
        self.import_builtin_components()
        try:
            return self._store[kind][name]
        except KeyError as exc:
            choices = sorted(self._store.get(kind, {}))
            raise KeyError(f"Unknown {kind} {name!r}. Available: {choices}") from exc

    def available(self, kind: str | None = None) -> dict[str, list[str]] | list[str]:
        self.import_builtin_components()
        if kind is None:
            return {k: sorted(v) for k, v in sorted(self._store.items())}
        return sorted(self._store.get(kind, {}))

    # ------------------------------------------------------------------ factory
    def build(self, kind: str, spec: dict[str, Any] | str | None, **extra: Any) -> Any:
        """Instantiate a component from a config spec.

        ``spec`` may be a bare name string or a dict ``{"name": ..., "params": {...}}``.
        Extra keyword arguments are forwarded to the constructor (e.g. shared feature
        specs) and take precedence over ``params``.
        """
        if spec is None:
            raise ValueError(f"No spec provided for kind={kind!r}")
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            name = spec.get("name") or spec.get("type")
            if name is None:
                raise ValueError(f"Spec for {kind} must contain 'name': {spec}")
            params = dict(spec.get("params") or {})
        params.update(extra)
        cls = self.get(kind, name)
        logger.debug("Building %s/%s with params=%s", kind, name, params)
        return cls(**params)

    # ------------------------------------------------------------------ builtins
    def import_builtin_components(self, modules: Iterable[str] = BUILTIN_MODULES) -> None:
        if self._builtins_loaded:
            return
        self._builtins_loaded = True
        for mod in modules:
            try:
                importlib.import_module(mod)
            except ModuleNotFoundError as exc:
                # Module not created yet (phased development) or optional dependency absent.
                logger.debug("Skipping builtin module %s: %s", mod, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed importing builtin module %s: %s", mod, exc)


_GLOBAL = Registry()

register = _GLOBAL.register
get = _GLOBAL.get
available = _GLOBAL.available
build = _GLOBAL.build
import_builtin_components = _GLOBAL.import_builtin_components
