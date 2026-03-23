from __future__ import annotations

import platform
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import psutil


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def get_hardware_snapshot() -> dict[str, object]:
    mem = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": round(mem.total / (1024**2), 2),
    }


def get_environment_snapshot() -> dict[str, object]:
    return {
        "python_version": sys.version,
        "git_commit": get_git_commit(),
        "hardware": get_hardware_snapshot(),
    }
