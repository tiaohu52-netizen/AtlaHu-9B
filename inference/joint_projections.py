"""Evaluate sibling projections of the same token in one existing W8A16 GEMV."""

from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps

import torch


FAMILIES = {
    "linear_attention": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    "attention": ("q_proj", "k_proj", "v_proj"),
    "mlp": ("gate_proj", "up_proj"),
}


def _replace(scope, owner, forward):
    if "forward" in vars(owner):
        scope.callback(setattr, owner, "forward", owner.forward)
    else:
        scope.callback(delattr, owner, "forward")
    owner.forward = forward


def _bind(owner, names, scope):
    from atlas_runtime.cuda_w8a16 import fused_gemv
    children = [getattr(owner, name) for name in names]
    if any(not hasattr(c, "_atlas_wq") or c.bias is not None for c in children):
        return False
    widths = {(c._atlas_wq.shape[1], c._atlas_group) for c in children}
    if len(widths) != 1:
        return False
    width, group = widths.pop()
    table = getattr(children[0], "_region_table", None)
    if any(getattr(child, "_region_table", None) is not table for child in children):
        return False
    lengths = [c._atlas_wq.shape[0] for c in children]
    # The existing kernel tiles output rows independently. Avoid padding at a sibling boundary.
    if any(n % 16 for n in lengths):
        return False
    q = torch.cat([c._atlas_wq for c in children], dim=0)
    scale = torch.cat([c._atlas_scale for c in children], dim=0)
    for child, q_slice, s_slice in zip(children, q.split(lengths), scale.split(lengths)):
        child._atlas_wq, child._atlas_scale = q_slice, s_slice
    values = ContextVar("sibling_projection_results", default=None)
    original = owner.forward

    @wraps(original)
    def forward(*args, **kwargs):
        x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
        if (not torch.is_tensor(x) or x.shape != (1, 1, width) or x.dtype != torch.bfloat16
                or torch.is_grad_enabled()):
            return original(*args, **kwargs)
        if table is None:
            joined = fused_gemv(x, q, scale, q.shape[0], group=group)
        else:
            from region_codec import gemv
            joined = gemv(x, q, scale, table, group, children[0]._region_bits)
        token = values.set((x, joined.split(lengths, dim=-1)))
        try:
            return original(*args, **kwargs)
        finally:
            values.reset(token)

    _replace(scope, owner, forward)
    for i, child in enumerate(children):
        old = child.forward

        def lookup(input, _i=i, _old=old):
            shared = values.get()
            return shared[1][_i] if shared is not None and shared[0] is input else _old(input)

        _replace(scope, child, lookup)
    return True


@contextmanager
def joint_projections(model, families):
    counts = Counter()
    with ExitStack() as scope:
        for owner in model.modules():
            for family in families:
                names = FAMILIES[family]
                if all(hasattr(owner, name) for name in names) and _bind(owner, names, scope):
                    counts[family] += 1
        yield dict(counts)
