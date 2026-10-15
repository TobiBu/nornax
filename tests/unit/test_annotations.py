"""Every annotation in the package resolves.

The modules use ``from __future__ import annotations``, so a name missing from a
module's imports is not an import error: the annotation stays a string until
something resolves it -- ``typing.get_type_hints``, a documentation build, or a
runtime type checker reaching that parameter. ``shooting_node`` annotated
``base_index`` with ``IntScalar`` without importing it, and the beartype smoke
run (``NORNAX_RUNTIME_TYPECHECK=1``) never resolved it.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing

import nornax


def _annotated_objects():
    """Yield ``(qualified name, object)`` for every function and class defined in nornax."""
    for info in pkgutil.walk_packages(nornax.__path__, "nornax."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{obj.__qualname__}", obj
            elif inspect.isclass(obj):
                yield f"{module.__name__}.{obj.__qualname__}", obj
                for member in vars(obj).values():
                    if inspect.isfunction(member):
                        yield f"{module.__name__}.{member.__qualname__}", member


def test_every_annotation_resolves() -> None:
    """``typing.get_type_hints`` succeeds on every function and class in the package."""
    unresolved = []
    for name, obj in _annotated_objects():
        try:
            typing.get_type_hints(obj)
        except NameError as exc:
            unresolved.append(f"{name}: {exc}")
    assert not unresolved, "\n".join(unresolved)
