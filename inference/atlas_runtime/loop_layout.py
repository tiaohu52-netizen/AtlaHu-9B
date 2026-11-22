"""Bind the declared loop to the repeated physical layers on the loaded instance."""

from contextlib import contextmanager
from copy import deepcopy
import sys

from .extensions import installed


def loop_plans(model):
    """Resolve a model-supplied or installed loop planner, then the bundled provider."""
    declared = getattr(model, "atlas_loop_plans", None)
    if callable(declared):
        return declared()
    matches = []
    for entry in installed("atlas.loop_plans").values():
        planner = entry.load()
        if planner.supports(model):
            matches.append((entry.name, planner))
    if len(matches) > 1:
        raise ValueError(f"multiple loop planners match this model: {[name for name, _ in matches]}")
    return matches[0][1](model) if matches else FixedLoopPlans(model)


def loop_frame(model):
    matches = [m for m in model.modules() if hasattr(m, "_loop_plan") and hasattr(m, "layers")]
    if len(matches) != 1:
        raise NotImplementedError("needs_backend: expected one identifiable repeated decoder")
    text = matches[0]
    provider = sys.modules[type(text).__module__]
    raw = model.config.to_dict()
    settings = raw.get("loop_settings", {})
    first, end, visits = (settings.get(k) for k in ("start", "end", "eval_count"))
    if (
        not callable(getattr(provider, "execution_plan", None))
        or any(type(v) is not int for v in (first, end, visits))
        or not 0 <= first < end <= len(text.layers)
        or visits < 2
    ):
        raise NotImplementedError("needs_backend: invalid repeated decoder boundaries")
    plan = tuple(text._loop_plan)
    expected = (
        tuple(range(first))
        + tuple(range(first, end)) * visits
        + tuple(range(end, len(text.layers)))
    )
    if tuple(provider.execution_plan(raw)) != plan or plan != expected:
        raise ValueError("declared loop does not match the resident physical execution order")
    return text, provider, raw


class FixedLoopPlans:
    """Fixed decoder execution plans and matching virtual-cache configurations."""

    def profile_modules(self, model):
        """Historical timing groups of this provider's repeated decoder."""
        import torch

        groups = {}
        for layer in self.text.layers:
            for attr, group in (("mhc_attn", "router"), ("mhc_ffn", "router"),
                                ("mlp", "ffn"), ("linear_attn", "linear_attention"),
                                ("self_attn", "full_attention"), ("input_layernorm", "layer_norm"),
                                ("post_attention_layernorm", "layer_norm")):
                module = getattr(layer, attr, None)
                if isinstance(module, torch.nn.Module):
                    groups[module] = group
            for name in ("gate_proj", "up_proj", "down_proj"):
                module = getattr(getattr(layer, "mlp", None), name, None)
                if isinstance(module, torch.nn.Module):
                    groups[module] = "ffn." + name
        groups[self.text.norm] = "layer_norm"
        groups[model.get_output_embeddings()] = "vocabulary_head"
        return groups

    def __init__(self, model):
        modules = [
            m
            for m in model.modules()
            if hasattr(m, "_loop_plan") and hasattr(m, "_loop_cache_config")
        ]
        if len(modules) != 1:
            raise NotImplementedError(
                "needs_backend: expected one decoder exposing a loop plan and virtual cache"
            )
        self.text = modules[0]
        self.provider = sys.modules[type(self.text).__module__]
        self.raw = model.config.to_dict()
        setting = self.raw.get("loop_settings", {})
        self.full = setting.get("eval_count")
        if (
            not callable(getattr(self.provider, "execution_plan", None))
            or type(self.full) is not int
            or self.full < 2
        ):
            raise NotImplementedError(
                "needs_backend: checkpoint has no verifiable repeated execution plan"
            )
        baseline = self.provider.execution_plan(self.raw)
        if tuple(baseline) != tuple(self.text._loop_plan):
            raise ValueError("resident execution differs from checkpoint declaration")
        self.first, self.last = setting["start"], setting["end"] - 1
        self.counts = list(range(1, self.full + 1))
        self.plans = {}
        for count in self.counts:
            config = deepcopy(self.raw)
            config["loop_settings"].update(eval_count=count, counts=[count])
            plan = self.provider.execution_plan(config)
            cache = deepcopy(self.text.config)
            cache.num_hidden_layers = len(plan)
            cache.layer_types = [self.text.config.layer_types[i] for i in plan]
            self.plans[count] = (plan, cache)

    @contextmanager
    def plan(self, count):
        original = self.text._loop_plan, self.text._loop_cache_config
        self.text._loop_plan, self.text._loop_cache_config = self.plans[count]
        try:
            yield
        finally:
            self.text._loop_plan, self.text._loop_cache_config = original
