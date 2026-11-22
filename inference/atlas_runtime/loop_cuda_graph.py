"""Capture measured fixed-size regions of a loaded model, retaining normal prefill and observable hooks."""

import hashlib
import inspect
import threading
from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps

import torch

from .loop_graph_inputs import GraphUnsupported, Inputs, recurrent_slot


def graph_regions(model, plans):
    paths = {m: n for n, m in model.named_modules()}
    regions = {}
    for i, layer in enumerate(plans.text.layers):
        linear = getattr(layer, "linear_attn", None)
        if linear is not None:
            # Physical boundaries execute the current handoff's Python control on each visit.
            owner = linear if i in (plans.first, plans.last) else layer
            regions[paths[owner]] = (
                "linear",
                "cache_params" if owner is linear else "past_key_values",
            )
            if owner is layer:
                continue
        for name in ("mhc_attn", "mhc_ffn", "input_layernorm", "post_attention_layernorm", "mlp"):
            module = getattr(layer, name, None)
            if module in paths:
                regions[paths[module]] = ("mlp" if name == "mlp" else "residual", None)
    regions[paths[plans.text.norm]] = ("residual", None)
    return regions


def source_digest(module):
    return hashlib.sha256(
        inspect.getsource(inspect.unwrap(type(module).forward)).encode("utf8")
    ).hexdigest()


def arithmetic_mode():
    return (
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
    )


def operand_versions(module):
    # Storage replacement (including a device move) invalidates a recording. In-place writes are read by the kernels.
    return tuple(
        (id(t), t.data_ptr(), t.dtype, tuple(t.shape), tuple(t.stride()))
        for t in (*module.parameters(), *module.buffers())
    )


class Recording:
    def __init__(self, original, args, kwargs, cache_arg, stream, counters, warmups):
        self.inputs = Inputs(args, kwargs, cache_arg)
        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.current_stream()
        stream.wait_stream(self.stream)
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                self.inputs.load(args, kwargs)
                original(*self.inputs.args, **self.inputs.kwargs)
        self.stream.wait_stream(stream)
        self.inputs.load(args, kwargs)
        expected = original(*self.inputs.args, **self.inputs.kwargs)
        expected_outputs = [
            v.clone() for v in ((expected,) if torch.is_tensor(expected) else expected)
        ]
        expected_states = (
            [v.clone() for v in self.inputs.cache.buffers] if self.inputs.cache else []
        )
        self.inputs.load(args, kwargs)
        before = [dict(c) for c in counters]
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = original(*self.inputs.args, **self.inputs.kwargs)
        if not (
            torch.is_tensor(self.output)
            or (isinstance(self.output, tuple) and all(torch.is_tensor(v) for v in self.output))
        ):
            raise GraphUnsupported("graph output is not a tensor or a tuple of tensors")
        self.increments = []
        for counter, old in zip(counters, before, strict=True):
            delta = {
                key: value - old.get(key, 0)
                for key, value in counter.items()
                if value != old.get(key, 0)
            }
            # Capture recorded these launches without executing them.
            counter.subtract(delta)
            if delta:
                self.increments.append((counter, delta))
        self.graph.replay()
        for counter, delta in self.increments:
            counter.update(delta)
        outputs = (self.output,) if torch.is_tensor(self.output) else self.output
        states = self.inputs.cache.buffers if self.inputs.cache else ()
        if not all(
            torch.equal(a, b)
            for a, b in zip((*outputs, *states), (*expected_outputs, *expected_states), strict=True)
        ):
            raise GraphUnsupported(
                "capture replay changed the region output or complete recurrent state"
            )

    def replay(self, args, kwargs):
        actual = self.inputs.load(args, kwargs)
        self.graph.replay()
        for counter, delta in self.increments:
            counter.update(delta)
        return self.inputs.result(actual, self.output)


class Region:
    def __init__(self, module, family, cache_arg, engine):
        self.module, self.family, self.cache_arg, self.engine = module, family, cache_arg, engine
        self.original = module.forward
        self.descendants = tuple(m for m in module.modules() if m is not module)
        self.lock = threading.Lock()
        self.graph = None
        self.versions = None
        self.failed = None
        self.calls = Counter()

    def counted(self, name):
        self.calls[name] += 1
        self.engine.calls[name] += 1

    def invoke(self, *args, **kwargs):
        graph = self.engine
        if self.family not in graph.active or self.module.training or torch.is_grad_enabled():
            return self.original(*args, **kwargs)
        h = args[0] if args else kwargs.get("hidden_states")
        if not torch.is_tensor(h) or not h.is_cuda or h.ndim < 3 or tuple(h.shape[:2]) != (1, 1):
            return self.original(*args, **kwargs)
        if (
            torch.cuda.is_current_stream_capturing()
            or torch.is_autocast_enabled("cuda")
            or arithmetic_mode() != graph.arithmetic
        ):
            return self.original(*args, **kwargs)
        if self.cache_arg:
            if kwargs.get(self.cache_arg) is None:
                return self.original(*args, **kwargs)
            try:
                recurrent_slot(kwargs[self.cache_arg])
            except GraphUnsupported:
                return self.original(*args, **kwargs)
        from torch.nn.modules import module as hooks

        if (
            hooks._global_forward_pre_hooks
            or hooks._global_forward_hooks
            or any(m._forward_hooks or m._forward_pre_hooks for m in self.descendants)
        ):
            self.counted("observed_eager")
            return self.original(*args, **kwargs)
        if self.failed or not self.lock.acquire(blocking=False):
            return self.original(*args, **kwargs)
        try:
            with torch.cuda.device(h.device):
                current = torch.cuda.current_stream()
                if self.graph is not None and current != self.graph.stream:
                    return self.original(*args, **kwargs)
                versions = operand_versions(self.module)
                if self.graph is None or versions != self.versions:
                    try:
                        self.graph = Recording(
                            self.original,
                            args,
                            kwargs,
                            self.cache_arg,
                            graph.capture_stream,
                            graph.counters,
                            graph.warmups,
                        )
                        self.versions = versions
                        self.counted("captures")
                    except (GraphUnsupported, RuntimeError) as exc:
                        if not isinstance(exc, GraphUnsupported) and not any(
                            part in str(exc).lower() for part in ("captur", "cuda graph")
                        ):
                            raise
                        self.failed = str(exc)
                        self.counted("unavailable")
                        self.graph = None
                        return self.original(*args, **kwargs)
                try:
                    result = self.graph.replay(args, kwargs)
                except GraphUnsupported:
                    self.counted("layout_eager")
                    return self.original(*args, **kwargs)
                self.counted("replays")
                return result
        finally:
            self.lock.release()


class GraphExecution:
    def __init__(self, loaded, plans, warmups, operators=None):
        self.model, self.plans, self.warmups = loaded.model, plans, warmups
        self.arithmetic = arithmetic_mode()
        self.layout = graph_regions(loaded.model, plans)
        self.capture_stream = torch.cuda.Stream(device=plans.text.embed_tokens.weight.device)
        if operators is None:
            operators = loaded.meta.get("loop_execution", {}).get("operators", {})
        self.calls = Counter()
        self.counters = [
            v["calls"] for v in operators.values() if isinstance(v.get("calls"), Counter)
        ]
        self.regions = {}
        self.active = set()

    @contextmanager
    def mount(self, paths):
        with ExitStack() as scope:
            for name in paths:
                module = self.model.get_submodule(name)
                family, cache_arg = self.layout[name]
                holder = Region(module, family, cache_arg, self)
                if "forward" in vars(module):
                    scope.callback(setattr, module, "forward", module.forward)
                else:
                    scope.callback(delattr, module, "forward")

                @wraps(holder.original)
                def invoke(*args, _holder=holder, **kwargs):
                    return _holder.invoke(*args, **kwargs)

                module.forward = invoke
                self.regions[name] = holder
            try:
                yield self
            finally:
                self.regions.clear()

    def evidence(self):
        return {
            n: dict(family=r.family, calls=dict(r.calls), unavailable=r.failed)
            for n, r in self.regions.items()
        }
