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
