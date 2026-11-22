"""CUDA inference specialization of finite FP32 row/column normalization on 2×2 matrices."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _logadd(x, y):
    m = tl.maximum(x, y)
    return m + libdevice.log(libdevice.exp(x - m) + libdevice.exp(y - m))


@triton.jit
def _finite(src, dst, count, stride: tl.constexpr, iterations: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = row < count
    offset = row * stride
    a = tl.load(src + offset, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(src + offset + 1, mask=valid, other=0.0).to(tl.float32)
    c = tl.load(src + offset + 2, mask=valid, other=0.0).to(tl.float32)
    d = tl.load(src + offset + 3, mask=valid, other=0.0).to(tl.float32)
    for _ in range(iterations):
        u = _logadd(a, b)
        v = _logadd(c, d)
        a, b = a - u, b - u
        c, d = c - v, d - v
        u = _logadd(a, c)
        v = _logadd(b, d)
        a, c = a - u, c - u
        b, d = b - v, d - v
    tl.store(dst + row * 4, libdevice.exp(a), mask=valid)
    tl.store(dst + row * 4 + 1, libdevice.exp(b), mask=valid)
    tl.store(dst + row * 4 + 2, libdevice.exp(c), mask=valid)
    tl.store(dst + row * 4 + 3, libdevice.exp(d), mask=valid)


def finite_2x2(logits, iterations):
    rows = logits.reshape(-1, 4)
    if rows.stride(1) != 1:
        rows = rows.contiguous()
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    with torch.cuda.device(logits.device):
        _finite[(triton.cdiv(len(rows), 32),)](
            rows,
            out,
            len(rows),
            rows.stride(0),
            iterations,
            32,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out
