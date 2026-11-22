"""Instance-local rebinding of a helper used by a Torch module forward."""

from contextlib import ExitStack, contextmanager
from functools import update_wrapper
from types import FunctionType, MethodType


def _cell(value):
    return (lambda: value).__closure__[0]


def _with_helper(fn, symbol, original, replacement):
    if not isinstance(fn, FunctionType):
        return None
    globals_ = fn.__globals__
    closure = fn.__closure__
    wrapped = getattr(fn, "__wrapped__", None)
    child = None
    if symbol in fn.__code__.co_names and globals_.get(symbol) is original:
        globals_ = {**globals_, symbol: replacement}
    elif closure and wrapped is not None and any(c.cell_contents is wrapped for c in closure):
        child = _with_helper(wrapped, symbol, original, replacement)
        if child is None:
            return None
        closure = tuple(_cell(child) if c.cell_contents is wrapped else c for c in closure)
    else:
        return None
    new = FunctionType(fn.__code__, globals_, fn.__name__, fn.__defaults__, closure)
    new.__kwdefaults__ = fn.__kwdefaults__
    update_wrapper(new, fn)
    if child is not None:
        new.__wrapped__ = child
    return new


@contextmanager
def bind_helper(model, provider, symbol, replacement, *, original=None):
    """Bind the original forward code with one alternate helper on reached instances only."""
    if original is None:
        original = getattr(provider, symbol)
    count = 0
    with ExitStack() as scope:
        for module in model.modules():
            method = module.forward
            fn = getattr(method, "__func__", None)
            new = _with_helper(fn, symbol, original, replacement)
            if new is None:
                continue
            if "forward" in vars(module):
                scope.callback(setattr, module, "forward", method)
            else:
                scope.callback(delattr, module, "forward")
            module.forward = MethodType(new, module)
            count += 1
        yield count

"""Program the residual streams handed from one visit of a physical loop to the next."""

import sys
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps

import torch




class StreamHandoff(torch.nn.Module):
    """Three stored scalars control stream carry, signed mean shift and residual extrapolation."""

    def __init__(self, like, coefficients):
        super().__init__()
        self.coefficients = torch.nn.Parameter(like.new_tensor(coefficients), requires_grad=False)

    def forward(self, mean, entry, streams):
        carry, shift, relax = self.coefficients.float().unbind()
        center = mean.float()
        residual = streams.float()
        difference = (residual[..., 0, :] - residual[..., 1, :]) * 0.5
        shared = shift * difference + relax * (center - entry.float())
        value = torch.lerp(center.unsqueeze(-2), residual, carry) + shared.unsqueeze(-2)
        return value.to(mean.dtype)


def endpoints(text, first, last):
    provider = sys.modules[type(text).__module__]
    for layer, i in ((text.layers[first], first), (text.layers[last], last)):
        physical = getattr(layer, "_mhc_physical", None)
        settings = getattr(layer, "_mhc_settings", {})
        if physical != i or (
            settings.get("start"),
            settings.get("end"),
            settings.get("streams"),
        ) != (first, last + 1, 2):
            raise NotImplementedError(
                "needs_backend: the two residual-stream boundaries must coincide with the physical loop"
            )
    return provider


@contextmanager
def handoff(text, first, last, operator):
    provider = endpoints(text, first, last)
    enter, leave = provider.enter, provider.leave
    current = ContextVar("atlas_stream_handoff", default=None)
    calls = {"transfers": 0}

    def start(hidden, physical, settings):
        state = current.get()
        if state is None or state["count"] <= 1:
            return enter(hidden, physical, settings)
        state["visit"] += 1
        if state["pending"] is not None:
            value = operator(hidden, state["entry"], state["pending"])
            state["pending"] = None
            calls["transfers"] += 1
        else:
            value = enter(hidden, physical, settings)
        state["entry"] = hidden
        return value

    def finish(streams, physical, settings):
        state = current.get()
        if state is not None and 0 < state["visit"] < state["count"]:
            state["pending"] = streams
        return leave(streams, physical, settings)

    original = text.forward

    @wraps(original)
    def forward(*args, **kwargs):
        count = text._loop_plan.count(first)
        mark = current.set({"count": count, "visit": 0, "pending": None, "entry": None})
        try:
            return original(*args, **kwargs)
        finally:
            current.reset(mark)

    with ExitStack() as scope:
        a = scope.enter_context(bind_helper(text.layers[first], provider, "enter", start))
        z = scope.enter_context(bind_helper(text.layers[last], provider, "leave", finish))
        if (a, z) != (1, 1):
            raise NotImplementedError(
                "needs_backend: expected one entrance and one exit implementation"
            )
        if "forward" in vars(text):
            scope.callback(setattr, text, "forward", original)
        else:
            scope.callback(delattr, text, "forward")
        text.forward = forward
        yield calls


def install(text, config):
    """Construct a regular HF child module before the checkpoint loader restores its parameters."""
    if config["format"] != "atlas.stream_handoff/1":
        raise ValueError("unknown stream handoff")
    first, last = config["first_layer"], config["last_layer"]
    operator = StreamHandoff(text.embed_tokens.weight, [0.0, 0.0, 0.0])
    text.add_module("atlas_handoff", operator)
    scope = ExitStack()
    calls = scope.enter_context(handoff(text, first, last, operator))
    text._atlas_handoff_context = scope
    text._atlas_handoff_calls = calls

"""Specialize exact stream carry without introducing a parameter or changing an existing tensor."""

from contextlib import ExitStack

import torch




class CarryStreams(torch.nn.Module):
    def forward(self, mean, entry, streams):
        return streams


def install_program(text, config):
    mode = config["mode"]
    if mode == "carry":
        op = CarryStreams()
        text.add_module("atlas_handoff", op)
        scope = ExitStack()
        text._atlas_handoff_calls = scope.enter_context(
            handoff(text, config["first_layer"], config["last_layer"], op)
        )
        text._atlas_handoff_context = scope
    elif mode == "coefficients":
        install(text, config)
    else:
        raise ValueError("unknown stream handoff specialization")
    text._atlas_handoff_program_id = config["id"]
