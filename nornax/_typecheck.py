"""Runtime type-checking opt-in for nornax.

Set the environment variable ``NORNAX_RUNTIME_TYPECHECK=1`` before importing
``nornax`` to enable package-wide ``jaxtyping`` + ``beartype`` instrumentation.
"""

from __future__ import annotations

import os


def enable_runtime_typecheck() -> None:
    """Activate jaxtyping + beartype import hook when requested via env var.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if os.environ.get("NORNAX_RUNTIME_TYPECHECK", "0") == "1":
        try:
            from jaxtyping import install_import_hook

            install_import_hook("nornax", "beartype.beartype")
        except ImportError:
            pass  # pragma: no cover
