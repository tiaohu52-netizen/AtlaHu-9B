"""Bind a measured graph recipe to the original module paths on a normal Atlas model load."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

from .loop_cuda_graph import GraphExecution, source_digest


def implementation_digest(declaration=None):
    h = hashlib.sha256()
    files = ["loop_cuda_graph.py", "loop_graph_inputs.py"]
    if declaration is not None:
        files.extend(("loop_graph_declared.py", "loop_graph_profile.py"))
    for name in files:
        h.update(name.encode())
        h.update(Path(__file__).with_name(name).read_text(encoding="utf8").encode("utf8"))
    return h.hexdigest()


def graph_engine(loaded, plans, warmups, declaration, device, operators=None):
    if declaration is None:
        from .loop_layout import FixedLoopPlans
        if not isinstance(plans, FixedLoopPlans):
            raise NotImplementedError("needs_backend: supply capture_regions for this decoder")
        return GraphExecution(loaded, plans, warmups, operators)
    from .loop_graph_declared import DeclaredGraphExecution
    return DeclaredGraphExecution(loaded, plans, warmups, operators, declaration, device)


def validate_recipe(model, plans, engine, recipe):
    if list(engine.arithmetic) != recipe["arithmetic"]:
        raise ValueError("CUDA graph arithmetic settings differ from the recorded device execution")
    if getattr(plans.text, "_atlas_handoff_program_id", None) != recipe["program_id"]:
        raise ValueError("capture recipe is bound to a different loop computation")
    paths = recipe["regions"]
    if not paths or len(set(paths)) != len(paths) or set(paths) != set(recipe["module_sources"]):
        raise ValueError("capture requires a measured source for each distinct module")
    groups = {engine.layout[n][0] for n in paths if n in engine.layout}
    if groups != set(recipe["groups"]) or any(n not in engine.layout for n in paths):
        raise ValueError("measured execution regions do not belong to this layout")
    for n in paths:
        if source_digest(model.get_submodule(n)) != recipe["module_sources"][n]:
            raise ValueError("captured module source differs at " + n)


@contextmanager
def installed_graph(loaded, plans, recipe, operators):
    declaration = recipe.get("capture_regions")
    device = loaded.meta.get("device")
    if declaration is not None and device is None:
        device = next(loaded.model.parameters()).device
    engine = graph_engine(loaded, plans, recipe["capture_warmups"], declaration, device, operators)
    validate_recipe(loaded.model, plans, engine, recipe)
    families = set(recipe["groups"])
    engine.active = families
    engine.suspended = False

    def select_program(module, args):
        current = getattr(plans.text, "_atlas_handoff_program_id", None)
        engine.active = (
            families if not engine.suspended and current == recipe["program_id"] else set()
        )

    with engine.mount(recipe["regions"]):
        hook = plans.text.register_forward_pre_hook(select_program)
        loaded.graph_execution = engine
        try:
            yield {"calls": engine.calls, "bound_modules": len(engine.regions)}
        finally:
            hook.remove()
            del loaded.graph_execution


@contextmanager
def suspend_graph(loaded):
    engine = getattr(loaded, "graph_execution", None)
    if engine is None:
        yield False
        return
    active, suspended = engine.active, engine.suspended
    engine.suspended, engine.active = True, set()
    try:
        yield True
    finally:
        engine.suspended, engine.active = suspended, active


def invalidate_writes(engine, receipts):
    if engine is None:
        return
    targets = {r["parameter"] for r in receipts}
    for path, region in engine.regions.items():
        if region.graph is not None and any(name.startswith(path + ".") for name in targets):
            region.graph = None
            region.versions = None
            region.counted("weight_invalidations")
