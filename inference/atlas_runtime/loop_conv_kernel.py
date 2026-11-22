"""Bind and verify the one-token conv update on the resident linear-attention modules."""

import hashlib
import inspect
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from .cuda_conv_update import fused_conv_update
from .loop_operator_binding import bind_helper


def owner(plans):
    modules = [layer.linear_attn for layer in plans.text.layers if hasattr(layer, "linear_attn")]
    owners = {sys.modules[inspect.unwrap(type(m).forward).__module__] for m in modules}
    if len(owners) != 1:
        raise NotImplementedError("needs_backend: one identifiable linear-attention implementation")
    return owners.pop()


def conv_recipe(plans):
    mod = owner(plans)
    symbol = "causal_conv1d_update"
    return dict(
        implementation="fused_conv_update",
        mode="fused",
        reference_symbol=symbol,
        reference_sha256=hashlib.sha256(
            inspect.getsource(getattr(mod, symbol)).encode()
        ).hexdigest(),
        owner_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
    )


@contextmanager
def accelerated_conv(model, plans, recipe, check=False):
    provider = owner(plans)
    symbol = recipe["reference_symbol"]
    reference = getattr(provider, symbol)
    code = hashlib.sha256(inspect.getsource(reference).encode()).hexdigest()
    if (
        code != recipe["reference_sha256"]
        or hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()
        != recipe["owner_sha256"]
    ):
        raise ValueError("linear-attention implementation differs from the measured conv update")
    calls = Counter()
    evidence = dict(
        calls=calls, equal_output=True, equal_state=True, max_output_error=0.0, max_state_error=0.0
    )

    def operation(hidden_states, conv_state, weight, bias, activation=None, **kw):
        supported = (
            hidden_states.is_cuda
            and hidden_states.dim() == 3
            and hidden_states.shape[-1] == 1
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and bias is None
            and activation in (None, "silu")
            and conv_state.is_contiguous()
            and not (torch.is_grad_enabled() and hidden_states.requires_grad)
            and weight.shape[0] == conv_state.shape[1]
            and conv_state.shape[0] == hidden_states.shape[0]
        )
        if not supported:
            calls["fallback"] += 1
            return reference(hidden_states, conv_state, weight, bias, activation=activation, **kw)
        if check:
            saved = conv_state.clone()
        result = fused_conv_update(hidden_states, conv_state, weight, bias, activation)
        calls["accelerated"] += 1
        if check:
            original = reference(hidden_states, saved, weight, bias, activation=activation, **kw)
            # The reference consumed the saved window and slid the clone in place;
            # the fused kernel slid the real cache window. Compare both effects.
            evidence["equal_output"] &= torch.equal(result, original)
            evidence["equal_state"] &= torch.equal(conv_state, saved)
            evidence["max_output_error"] = max(
                evidence["max_output_error"],
                float((result.float() - original.float()).abs().amax()),
            )
            evidence["max_state_error"] = max(
                evidence["max_state_error"],
                float((conv_state.float() - saved.float()).abs().amax()),
            )
        return result

    previous = getattr(model, "_atlas_cached_conv", None)
    with bind_helper(
        model, provider, symbol, operation, original=previous[2] if previous else None
    ) as bound:
        if not bound:
            raise NotImplementedError("needs_backend: could not bind the actual conv update")
        evidence["bound_modules"] = bound
        model._atlas_cached_conv = (provider, symbol, operation)
        try:
            yield evidence
        finally:
            if previous is None:
                del model._atlas_cached_conv
            else:
                model._atlas_cached_conv = previous


@contextmanager
def provider_conv(model, plans):
    """Measure the provider conv while all other installed operators remain."""
    installed = getattr(model, "_atlas_cached_conv", None)
    if installed is None:
        yield False
        return
    provider, symbol, old = installed
    with bind_helper(model, provider, symbol, getattr(provider, symbol), original=old) as n:
        if not n:
            raise RuntimeError("installed conv update did not bind to the resident attention")
        del model._atlas_cached_conv
        try:
            yield True
        finally:
            model._atlas_cached_conv = installed
