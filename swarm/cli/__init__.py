"""Swarm Agent CLI module.

Provides the standalone command line interface, interactive setup wizard,
and daemon launcher for the Swarm Agent multi-provider gateway.
"""

from __future__ import annotations

__version__ = "1.0.0"


def __getattr__(name: str):
    if name in ("cli", "main"):
        from swarm.cli.client import cli, main
        if name == "cli":
            return cli
        return main
    if name == "run_wizard":
        from swarm.cli.wizard import run_wizard
        return run_wizard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cli", "main", "run_wizard", "__version__"]

