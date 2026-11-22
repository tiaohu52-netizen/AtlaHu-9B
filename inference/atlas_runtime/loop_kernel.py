"""Install a measured operator on individual router instances without changing the provider module."""

import hashlib
import inspect
import json
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from .loop_operator_binding import bind_helper


def implementation_identity(fn):
    return hashlib.sha256(inspect.getsource(fn).encode("utf8")).hexdigest()


def kernel_identity(implementation="finite_sinkhorn_2x2_fp32", recipe=None):
    from .loop_operator_registry import identity

    return identity(implementation, recipe)


def check_backend(device, provider, certificate):
    from .loop_operator_registry import lookup

    if certificate.get("accepted") is not True:
        raise ValueError("routing measurement did not accept this kernel")
    recipe = certificate["recipe"]
    operator = lookup(recipe["implementation"])
    evidence = certificate["provenance"]
    actual = {
        "device_type": torch.device(device).type,
        "torch_version": str(torch.__version__),
    }
    if operator.backend:
        actual.update(operator.backend(device, provider))
    actual.update(operator.identity(recipe))
    recorded = evidence.get("implementation_sha256", evidence.get(operator.legacy_digest))
    if not actual.get("implementation_sha256") or recorded != actual["implementation_sha256"]:
        raise ValueError("routing certificate differs from the active implementation_sha256")
    for k, v in actual.items():
        if k in evidence and evidence[k] != v:
            raise ValueError(f"routing certificate differs from the active {k}")
    for key, required in certificate.get("requires", {}).items():
        if key not in actual or actual[key] != required:
            raise ValueError(f"execution requirement differs or is unavailable: {key}")


def certified_binding(loaded, device, certificate, active):
    from .loop_operator_registry import lookup

    recipe = certificate["recipe"]
    operator = lookup(recipe["implementation"])
    plans = operator.plans(loaded.model) if operator.plans else None
    provider = getattr(plans, "provider", None) or inspect.getmodule(type(loaded.model))
    check_backend(device, provider, certificate)
    return operator.mount(loaded, plans, recipe, active)


@contextmanager
def accelerated(model, provider, recipe, check=False):
    if recipe["implementation"] != "finite_sinkhorn_2x2_fp32":
        raise ValueError("unknown loop kernel")
    from .cuda_sinkhorn import finite_2x2

    symbol = recipe["reference_symbol"]
    reference = getattr(provider, symbol)
    signature = implementation_identity(reference)
    if "reference_sha256" in recipe and signature != recipe["reference_sha256"]:
        raise ValueError("normalization helper differs from the measured implementation")
    calls = Counter()
    error = 0.0
    equal = True

    def normalize(logits, iterations):
        nonlocal error, equal
        if (
            not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.shape[-2:] != (2, 2)
            or not logits.numel()
            or iterations != recipe["iterations"]
            or torch.is_grad_enabled()
            and logits.requires_grad
        ):
            calls["fallback"] += 1
            return reference(logits, iterations)
        value = finite_2x2(logits, iterations)
        calls["accelerated"] += 1
        if check:
            orig = reference(logits, iterations)
            error = max(error, (orig - value).abs().max().item())
            equal = equal and torch.equal(orig, value)
        return value

    evidence = {"reference_sha256": signature, "calls": calls}
    with bind_helper(model, provider, symbol, normalize) as count:
        evidence["bound_modules"] = count
        try:
            yield evidence
        finally:
            evidence["max_router_error"] = error if check and calls["accelerated"] else None
            evidence["equal_router_outputs"] = equal if check and calls["accelerated"] else None
