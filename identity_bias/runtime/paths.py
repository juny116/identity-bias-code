"""Workspace paths independent of the Python source layout."""

import os
from pathlib import Path


def workspace_root():
    return (
        Path(os.environ.get("IDENTITY_BIAS_WORKSPACE", Path.cwd()))
        .expanduser()
        .resolve()
    )


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else workspace_root() / path
