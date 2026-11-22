"""Static CUDA graph inputs and private copies of an initialized virtual recurrent cache slot."""

from copy import copy
from collections.abc import Mapping

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class GraphUnsupported(Exception):
    """An invocation does not have the measured fixed, single-token execution semantics."""


def tensor_layout(t):
    return tuple(t.shape), tuple(t.stride()), t.dtype, t.device


def recurrent_slot(view):
    layers = getattr(view, "layers", None)
    physical = getattr(layers, "physical", None)
    if physical is None or not hasattr(view, "cache"):
        raise GraphUnsupported("no physical-to-virtual recurrent cache view")
    slot = layers[physical]
    if getattr(slot, "record_past", True) or not view.has_previous_state(physical, state_idx=0):
        raise GraphUnsupported("recurrent cache is recording history or not initialized")
    state = []
    for name in ("conv_states", "recurrent_states"):
        values = getattr(slot, name, None)
        if (
            not isinstance(values, (tuple, list, Mapping))
            or len(values) != 1
            or isinstance(values, Mapping)
            and 0 not in values
            or not torch.is_tensor(values[0])
        ):
            raise GraphUnsupported("expected one initialized convolution and recurrence state")
        state.append(values[0])
    return slot, tuple(state)


class PrivateCache:
    def __init__(self, view):
        slot, tensors = recurrent_slot(view)
        self.layout = tuple(tensor_layout(t) for t in tensors)
        self.buffers = tuple(t.detach().clone(memory_format=torch.preserve_format) for t in tensors)
        # Use the actual provider's slot and update methods on independent, fixed storage.
        private = copy(slot)
        for name, tensor in zip(("conv_states", "recurrent_states"), self.buffers, strict=True):
            original = getattr(slot, name)
            setattr(
                private,
                name,
                {0: tensor} if isinstance(original, Mapping) else type(original)([tensor]),
            )
        owner = copy(view.cache)
        owner.layers = [private]
        self.view = type(view)(owner, view.layers.physical, 0)
        self.slot_class = type(slot)

    def inputs(self, view):
        slot, actual = recurrent_slot(view)
        if (
            type(slot) is not self.slot_class
            or tuple(tensor_layout(t) for t in actual) != self.layout
        ):
            raise GraphUnsupported("recurrent cache physical layout changed")
        return actual

    def read(self, actual):
        for dst, src in zip(self.buffers, actual, strict=True):
            dst.copy_(src)

    def write(self, actual):
        for dst, src in zip(actual, self.buffers, strict=True):
            dst.copy_(src)


class Inputs:
    def __init__(self, args, kwargs, cache_arg):
        values = dict(kwargs)
        cache = values.pop(cache_arg) if cache_arg else None
        self.cache_arg = cache_arg
        self.cache = PrivateCache(cache) if cache is not None else None
        flat, self.spec = tree_flatten((args, values))
        self.static = []
        self.layout = []
        for v in flat:
            if torch.is_tensor(v):
                if not v.is_cuda or v.requires_grad and torch.is_grad_enabled():
                    raise GraphUnsupported("expected inference tensors on CUDA")
                self.static.append(v.detach().clone(memory_format=torch.preserve_format))
                self.layout.append(tensor_layout(v))
            elif v is None or type(v) in (bool, str, int, float):
                self.static.append(v)
                self.layout.append(None)
            else:
                raise GraphUnsupported("dynamic non-tensor argument " + type(v).__name__)
        self.args, self.kwargs = tree_unflatten(self.static, self.spec)
        if self.cache is not None:
            self.kwargs[cache_arg] = self.cache.view

    def load(self, args, kwargs):
        values = dict(kwargs)
        actual = self.cache.inputs(values.pop(self.cache_arg)) if self.cache is not None else None
        flat, spec = tree_flatten((args, values))
        if spec != self.spec:
            raise GraphUnsupported("call structure differs from the captured forward")
        for v, dst, layout in zip(flat, self.static, self.layout, strict=True):
            if layout is None:
                if type(v) is not type(dst) or v != dst:
                    raise GraphUnsupported("non-tensor execution argument changed")
            elif not torch.is_tensor(v) or tensor_layout(v) != layout:
                raise GraphUnsupported("tensor layout differs from the captured forward")
        for v, dst, layout in zip(flat, self.static, self.layout, strict=True):
            if layout is not None:
                dst.copy_(v)
        if actual is not None:
            self.cache.read(actual)
        return actual

    def result(self, actual, output):
        new = output.clone() if torch.is_tensor(output) else tuple(v.clone() for v in output)
        if actual is not None:
            self.cache.write(actual)
        return new
