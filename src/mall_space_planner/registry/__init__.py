"""Component registry and factory.

Every pluggable component (retriever, ranker, encoder, generator, ...) is registered
under a *kind* and a *name*::

    from mall_space_planner.registry import register, build

    @register("ranker", "knn")
    class KNNRanker(BaseRanker):
        ...

    ranker = build("ranker", {"name": "knn", "params": {"k": 20}})

Training / inference pipelines never import concrete classes; they only call
:func:`build` with the dictionary taken from YAML.
"""

from mall_space_planner.registry.registry import (
    Registry,
    available,
    build,
    get,
    import_builtin_components,
    register,
)

__all__ = ["Registry", "available", "build", "get", "import_builtin_components", "register"]
