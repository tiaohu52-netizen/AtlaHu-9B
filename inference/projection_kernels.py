"""One-pass reconstruction of grouped BF16 weights for multi-token forwards."""

from contextlib import contextmanager, ExitStack
import torch
import triton
import triton.language as tl


@triton.jit
def _materialize(q, scales, out, size, SCALE_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < size
    value = tl.load(q + index, mask=valid, other=0).to(tl.float32)
    scale = tl.load(scales + index // SCALE_WIDTH, mask=valid, other=1)
    tl.store(out + index, (value * scale).to(tl.bfloat16), mask=valid)


def materialize(q, scales, group):
    width = q.shape[1] if group == 1 else group
    result = torch.empty_like(q, dtype=torch.bfloat16)
    _materialize[(triton.cdiv(q.numel(), 512),)](q, scales, result, q.numel(), width, 512)
    return result


@contextmanager
def fast_prefill(model):
    """Retain cuBLAS arithmetic and the mounted one-token kernels."""
    counts = {"prefill_calls": 0}
    with ExitStack() as undo:
        for module in model.modules():
            if not hasattr(module, "_atlas_wq"):
                continue
            original = module.forward

            def forward(input, _m=module, _old=original):
                if input.numel() == input.shape[-1] or input.dtype != torch.bfloat16:
                    return _old(input)
                counts["prefill_calls"] += 1
                if hasattr(_m, "_region_table"):
                    from region_codec import materialize as indexed_weight
                    weight = indexed_weight(_m._atlas_wq, _m._atlas_scale, _m._region_table,
                                            _m._atlas_group, _m._region_bits)
                else:
                    weight = materialize(_m._atlas_wq, _m._atlas_scale, _m._atlas_group)
                return torch.nn.functional.linear(input, weight, _m.bias)

            undo.callback(setattr, module, "forward", original)
            module.forward = forward
        yield counts


@triton.jit
def _wide_gemv(x, q, scales, y, N: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < N) & (col[None, :] < K)
    codes = tl.load(q + rows[:, None] * K + col[None, :], mask=mask, other=0)
    scale = tl.load(scales + rows[:, None] * (K // GROUP) + col[None, :] // GROUP,
                    mask=mask, other=0)
    weight = (codes.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    activation = tl.load(x + col, mask=col < K, other=0).to(tl.float32)
    acc = tl.sum(weight * activation[None, :], axis=1)
    tl.store(y + rows, acc.to(tl.bfloat16), mask=rows < N)


def wide_gemv(x, q, scales, group, tile):
    n, k = q.shape
    out = x.new_empty((1, 1, n))
    rows, warps = tile
    _wide_gemv[(triton.cdiv(n, rows),)](
        x.contiguous(), q, scales, out, n, k, group, rows, triton.next_power_of_2(k), num_warps=warps
    )
    return out


@contextmanager
def selected_gemv(model, tiles):
    with ExitStack() as scope:
        for module in model.modules():
            q = getattr(module, "_atlas_wq", None)
            if q is None:
                continue
            key = f"{q.shape[0]}x{q.shape[1]}g{module._atlas_group}"
            if key not in tiles:
                continue
            original = module.forward

            def forward(input, _m=module, _old=original, _tile=tiles[key]):
                if input.shape == (1, 1, _m._atlas_wq.shape[1]) and input.dtype == torch.bfloat16 and _m.bias is None:
                    return wide_gemv(input, _m._atlas_wq, _m._atlas_scale, _m._atlas_group, _tile)
                return _old(input)

            scope.callback(setattr, module, "forward", original)
            module.forward = forward
        yield
