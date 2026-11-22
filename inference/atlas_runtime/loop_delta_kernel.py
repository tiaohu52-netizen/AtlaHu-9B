"""Bind and verify a one-token recurrence on individual linear-attention modules."""

import hashlib
import inspect
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from .loop_operator_binding import bind_helper


def owner(plans):
    modules = [layer.linear_attn for layer in plans.text.layers if hasattr(layer, "linear_attn")]
    owners = {sys.modules[inspect.unwrap(type(m).forward).__module__] for m in modules}
    if len(owners) != 1:
        raise NotImplementedError("needs_backend: one identifiable linear-attention implementation")
    return owners.pop()


@contextmanager
def provider_delta(model):
    """Measure the provider recurrence while retaining all other installed operators."""
    installed = getattr(model, "_atlas_cached_delta", None)
    if installed is None:
        yield False
        return
    provider, symbol, old = installed
    with bind_helper(model, provider, symbol, getattr(provider, symbol), original=old) as n:
        if not n:
            raise RuntimeError("installed recurrence did not bind to the resident attention")
        del model._atlas_cached_delta
        try:
            yield True
        finally:
            model._atlas_cached_delta = installed


@contextmanager
def accelerated_delta(model, plans, recipe, check=False):
    from .cuda_delta_rule import single_step

    provider = owner(plans)
    symbol = recipe["reference_symbol"]
    reference = getattr(provider, symbol)
    code = hashlib.sha256(inspect.getsource(reference).encode()).hexdigest()
    if (
        code != recipe["reference_sha256"]
        or hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()
        != recipe["owner_sha256"]
    ):
        raise ValueError("linear-attention implementation differs from the measured recurrence")
    mode = recipe["mode"]
    if mode not in ("split", "packed", "fused"):
        raise ValueError("unknown cached delta recurrence")
    calls = Counter()
    evidence = dict(
        calls=calls, equal_output=True, equal_state=True, max_output_error=0.0, max_state_error=0.0
    )

    def operation(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kw,
    ):
        supported = (
            query.is_cuda
            and query.ndim == 4
            and query.shape[1] == 1
            and query.shape == key.shape
            and query.shape[:3] == value.shape[:3]
            and initial_state is not None
            and tuple(initial_state.shape)
            == (query.shape[0], query.shape[2], key.shape[-1], value.shape[-1])
            and query.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and query.dtype == value.dtype
            and not (
                torch.is_grad_enabled()
                and any(x.requires_grad for x in (query, key, value, g, beta, initial_state))
            )
            and kw.get("cu_seqlens") is None
        )
        if not supported:
            calls["fallback"] += 1
            return reference(
                query,
                key,
                value,
                g,
                beta,
                initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                **kw,
            )
        result, state = single_step(
            provider, mode, query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel
        )
        calls["accelerated"] += 1
        if check:
            original, orig_state = reference(
                query, key, value, g, beta, initial_state, True, use_qk_l2norm_in_kernel, **kw
            )
            for n, x, y in (("output", result, original), ("state", state, orig_state)):
                evidence["equal_" + n] &= torch.equal(x, y)
                evidence["max_" + n + "_error"] = max(
                    evidence["max_" + n + "_error"], float((x.float() - y.float()).abs().amax())
                )
        return result, state if output_final_state else None

    previous = getattr(model, "_atlas_cached_delta", None)
    with bind_helper(
        model, provider, symbol, operation, original=previous[2] if previous else None
    ) as n:
        if not n:
            raise NotImplementedError("needs_backend: could not bind the actual cached delta rule")
        evidence["bound_modules"] = n
        model._atlas_cached_delta = (provider, symbol, operation)
        try:
            yield evidence
        finally:
            if previous is None:
                del model._atlas_cached_delta
            else:
                model._atlas_cached_delta = previous
