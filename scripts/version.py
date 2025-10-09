from __future__ import annotations

import os
import subprocess

from pdm.backend.hooks.version import SCMVersion


def _short_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def format_version(v: SCMVersion) -> str:
    # Base: exact tag on tagged commits, else add .post<distance>
    base = str(v.version) if v.distance is None else f"{v.version}.post{v.distance}"
    if False:
        sha = _short_sha()
        if sha:
            base = f"{base}+g{sha}"
    return base
