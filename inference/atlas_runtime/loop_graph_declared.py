"""CUDA capture on physical module paths explicitly selected for replay."""

from collections import Counter

import torch

from .loop_cuda_graph import GraphExecution, arithmetic_mode


def region_layout(model, declaration):
    if not isinstance(declaration, dict) or not declaration:
        raise ValueError("capture_regions needs a nonempty map of module paths")
    modules = dict(model.named_modules(remove_duplicate=False))
    layout = {}
    members = set()
    for path, spec in declaration.items():
        if not path or path not in modules or not isinstance(spec, dict):
            raise ValueError(f"unknown capture path or invalid declaration: {path!r}")
        family = spec.get("group")
        cache = spec.get("cache_argument")
        if (
            not isinstance(family, str)
            or not family.strip()
            or (cache is not None and (not isinstance(cache, str) or not cache))
        ):
            raise ValueError(f"invalid capture group or cache argument at {path!r}")
        descendants = set(modules[path].modules())
        if members.intersection(descendants):
            raise ValueError(f"overlapping capture regions at {path!r}")
        members.update(descendants)
        layout[path] = (family, cache)
    return layout


class DeclaredGraphExecution(GraphExecution):
    """Use the recorded replay engine with a caller-declared layout and execution device."""

    def __init__(self, loaded, plans, warmups, operators, declaration, device):
        self.model, self.plans, self.warmups = loaded.model, plans, warmups
        self.arithmetic = arithmetic_mode()
        self.layout = region_layout(self.model, declaration)
        self.capture_stream = torch.cuda.Stream(device=device)
        if operators is None:
            operators = loaded.meta.get("loop_execution", {}).get("operators", {})
        self.counters = [
            op["calls"]
            for op in operators.values()
            if isinstance(op.get("calls"), Counter)
        ]
        self.calls = Counter()
        self.regions = {}
        self.active = set()
