"""Discover extensions installed in the active Python environment without importing their code."""

from functools import cache, lru_cache
from importlib.metadata import entry_points


@cache
def installed(group):
    entries = {}
    for entry in entry_points(group=group):
        if entry.name in entries:
            raise ValueError(f"duplicate {group} entry point {entry.name!r}")
        entries[entry.name] = entry
    return entries
