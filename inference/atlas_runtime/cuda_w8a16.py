"""Fused int8-resident GEMV for single-token decode.

Certified value sets: symmetric int8 with ``amax`` scales — per output row, or
per 32-column group within a row — where ``bf16(Wq * scale)`` feeds the usual
BF16 GEMM. The kernel keeps exactly those value sets: dequantize per element to
BF16, multiply with the BF16 input, accumulate in FP32, round the output to
BF16 once. v2 removes the per-element integer division (the scale index is a
chunk-level scalar per group) and masks the K tail, so arbitrary input widths
like down_proj's 12096 take the fused path instead of the eager dequant
fallback.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_w8a16(
    x_ptr,
    wq_ptr,
    scale_ptr,
    y_ptr,
    IN: tl.constexpr,
    OUT,
    GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < OUT
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if GROUP == 1:
        for k in range(0, IN, BLOCK_K):
            ks = k + tl.arange(0, BLOCK_K)
            kmask = ks < IN
            x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
            wq = tl.load(
                wq_ptr + rows[:, None] * IN + ks[None, :],
                mask=live[:, None] & kmask[None, :],
                other=0,
            )
            scale = tl.load(scale_ptr + rows, mask=live, other=0.0).to(tl.float32)
            w = (wq.to(tl.float32) * scale[:, None]).to(tl.bfloat16).to(tl.float32)
            acc += tl.sum(x[None, :] * w, axis=1)
    else:
        for k in range(0, IN, BLOCK_K):
            for g in tl.static_range(0, BLOCK_K // GROUP):
                ks = k + g * GROUP + tl.arange(0, GROUP)
                kmask = ks < IN
                x = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
                wq = tl.load(
                    wq_ptr + rows[:, None] * IN + ks[None, :],
                    mask=live[:, None] & kmask[None, :],
                    other=0,
                )
                s = tl.load(
                    scale_ptr + rows * (IN // GROUP) + (k // GROUP + g),
                    mask=live,
                    other=0.0,
                ).to(tl.float32)
                w = (wq.to(tl.float32) * s[:, None]).to(tl.bfloat16).to(tl.float32)
                acc += tl.sum(x[None, :] * w, axis=1)
    tl.store(y_ptr + rows, acc.to(tl.bfloat16), mask=live)


def fused_gemv(x, wq, scale, out_features, group=1):
    """x: [1, 1, IN] bf16 contiguous; wq: [OUT, IN] int8; scale: [OUT] fp32 for
    per-row scales, or [OUT, IN/group] fp32 for grouped scales."""
    in_features = x.shape[-1]
    if in_features != wq.shape[1]:
        raise NotImplementedError("needs_backend: matching input width")
    if group != 1 and (scale.dim() != 2 or scale.shape[1] != in_features // group):
        raise NotImplementedError("needs_backend: grouped scales [OUT, IN/group]")
    if in_features % group:
        raise NotImplementedError("needs_backend: input width divisible by the group size")
    y = torch.empty((1, 1, out_features), dtype=torch.bfloat16, device=x.device)
    flat = x.reshape(1, in_features).contiguous()
    block_n = 16
    block_k = 512 if in_features % 512 == 0 else 128
    if block_k % group:
        block_k = group * max(1, (128 // group))
    with torch.cuda.device(x.device):
        _gemv_w8a16[(triton.cdiv(out_features, block_n),)](
            flat,
            wq,
            scale.contiguous(),
            y,
            in_features,
            out_features,
            group,
            block_k,
            block_n,
            num_warps=4,
        )
    return y


def dequantized(wq, scale, dtype=torch.bfloat16, group=1):
    """The exact BF16 value set the probe measured, materialized for reference paths."""
    if group == 1:
        return (wq.to(torch.float32) * scale[:, None]).to(dtype)
    expanded = scale.repeat_interleave(group, dim=1)
    return (wq.to(torch.float32) * expanded).to(dtype)


def implementation_identity():
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
