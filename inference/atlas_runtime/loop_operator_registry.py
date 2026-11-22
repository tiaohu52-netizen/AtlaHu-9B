"""Bindings for saved execution profiles, including installed atlas.loop_operators."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .extensions import installed
from .loop_layout import loop_plans


@dataclass(frozen=True)
class Operator:
    """identity(recipe) identifies code; mount(loaded, plans, recipe, active) owns cleanup."""

    identity: Callable
    mount: Callable
    plans: Callable | None = None
    backend: Callable | None = None
    legacy_digest: str | None = None


def _triton_identity(source, recipe):
    try:
        triton = version("triton")
    except PackageNotFoundError as exc:
        raise NotImplementedError("needs_backend: this kernel requires Triton") from exc
    return {
        "implementation_sha256": hashlib.sha256(
            Path(__file__).with_name(source).read_bytes()
        ).hexdigest(),
        "triton_version": triton,
    }


def _cuda(device, provider):
    import torch

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise NotImplementedError("needs_backend: this implementation requires CUDA")
    return {
        "capability": list(torch.cuda.get_device_capability(device)),
        "provider_sha256": hashlib.sha256(
            Path(provider.__file__).read_bytes()
        ).hexdigest(),
    }


def _sinkhorn(loaded, plans, recipe, active):
    from .loop_kernel import accelerated

    return accelerated(loaded.model, plans.provider, recipe)


def _delta(loaded, plans, recipe, active):
    from .loop_delta_kernel import accelerated_delta

    return accelerated_delta(loaded.model, plans, recipe)


def _conv(loaded, plans, recipe, active):
    from .loop_conv_kernel import accelerated_conv

    return accelerated_conv(loaded.model, plans, recipe)


def _graph_identity(recipe):
    from .loop_graph_profile import implementation_digest

    return {
        "implementation_sha256": implementation_digest(recipe.get("capture_regions"))
    }


def _graph(loaded, plans, recipe, active):
    from .loop_graph_profile import installed_graph

    return installed_graph(loaded, plans, recipe, active)


def _triton(source, mount, legacy_digest=None):
    return Operator(
        partial(_triton_identity, source), mount, loop_plans, _cuda, legacy_digest
    )


_BUILTINS = {
    "finite_sinkhorn_2x2_fp32": _triton("cuda_sinkhorn.py", _sinkhorn),
    "cached_gated_delta": _triton("cuda_delta_rule.py", _delta),
    "fused_conv_update": _triton(
        "cuda_conv_update.py", _conv, "conv_implementation_sha256"
    ),
    "cuda_graph_decode": Operator(_graph_identity, _graph, loop_plans, _cuda),
}


def lookup(name):
    point = installed("atlas.loop_operators").get(name)
    if point is not None:
        if name in _BUILTINS:
            raise ValueError(f"duplicate execution operator {name!r}")
        try:
            operator = point.load()
        except ImportError as exc:
            raise NotImplementedError(
                f"needs_backend: cannot import {point.value}: {exc}"
            ) from exc
        if not isinstance(operator, Operator):
            raise ValueError(f"{point.value} must export an Operator")
        return operator
    if name not in _BUILTINS:
        raise ValueError(f"unknown execution operator {name!r}")
    return _BUILTINS[name]


def identity(implementation, recipe=None):
    return lookup(implementation).identity(recipe or {})


def mount(loaded, plans, recipe, active=None):
    return lookup(recipe["implementation"]).mount(loaded, plans, recipe, active or {})
