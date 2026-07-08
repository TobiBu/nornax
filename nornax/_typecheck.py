"""Runtime type-checking opt-in for nornax.

Set the environment variable ``NORNAX_RUNTIME_TYPECHECK=1`` before importing
``nornax`` to enable package-wide ``jaxtyping`` + ``beartype`` instrumentation.

The jaxtyping import hook only instruments modules imported *after* it is
installed, so ``nornax.__init__`` calls :func:`enable_runtime_typecheck` before
importing any nornax submodules.
"""

from __future__ import annotations

import os


def enable_runtime_typecheck() -> None:
    """Install the jaxtyping + beartype import hook when requested via env var."""
    if os.environ.get("NORNAX_RUNTIME_TYPECHECK", "0") != "1":
        return
    try:
        from jaxtyping import install_import_hook
    except ImportError:  # pragma: no cover - jaxtyping is a hard dependency
        return
    install_import_hook("nornax", "beartype.beartype")
