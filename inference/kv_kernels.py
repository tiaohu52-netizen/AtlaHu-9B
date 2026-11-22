"""One CUDA launch per K/V pair. Each vector is independently scaled in FP32."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pack_one(source, destination, row, offsets, WIDTH: tl.constexpr):
    x = tl.load(source + row * WIDTH + offsets, mask=offsets < WIDTH, other=0).to(tl.float32)
    scale = tl.div_rn(tl.max(tl.abs(x), axis=0), 127.0)
    scale = tl.where(scale == 0, 1.0, scale)
    code = libdevice.rint(tl.div_rn(x, scale)).to(tl.int8)
    target = destination + row * (WIDTH + 4)
    tl.store(target + offsets, code.to(tl.uint8, bitcast=True), mask=offsets < WIDTH)
    tl.store((target + WIDTH).to(tl.pointer_type(tl.float32)), scale)


@triton.jit
def _pack_both(k, v, pk, pv, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    _pack_one(k, pk, row, offsets, WIDTH)
    _pack_one(v, pv, row, offsets, WIDTH)


@triton.jit
def _unpack_one(source, destination, row, offsets, WIDTH: tl.constexpr):
    address = source + row * (WIDTH + 4)
    scale = tl.load((address + WIDTH).to(tl.pointer_type(tl.float32)))
    code = tl.load(address + offsets, mask=offsets < WIDTH, other=0).to(tl.int8, bitcast=True)
    tl.store(destination + row * WIDTH + offsets, code.to(tl.float32) * scale, mask=offsets < WIDTH)


@triton.jit
def _unpack_both(pk, pv, k, v, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    _unpack_one(pk, k, row, offsets, WIDTH)
    _unpack_one(pv, v, row, offsets, WIDTH)


def pack_cuda(keys, values):
    width = keys.shape[-1]
    if keys.shape != values.shape or width % 4:
        raise ValueError("paired KV packing requires equal shapes and word-aligned head dimensions")
    shape = (*keys.shape[:-1], width + 4)
    packed = [torch.empty(shape, device=keys.device, dtype=torch.uint8) for _ in range(2)]
    grid = (keys.numel() // width,)
    _pack_both[grid](keys.contiguous(), values.contiguous(), *packed, width, triton.next_power_of_2(width))
    return tuple(packed)


def unpack_cuda(keys, values, dtype):
    width = keys.shape[-1] - 4
    shape = (*keys.shape[:-1], width)
    restored = [torch.empty(shape, device=keys.device, dtype=dtype) for _ in range(2)]
    grid = (keys.numel() // (width + 4),)
    _unpack_both[grid](keys.contiguous(), values.contiguous(), *restored, width, triton.next_power_of_2(width))
    return tuple(restored)


@triton.jit
def _append_vector(old, new, packed, restored, old_row, new_row, source_row,
                   is_new, offsets, WIDTH: tl.constexpr):
    target = packed + new_row * (WIDTH + 4)
    if is_new:
        x = tl.load(new + source_row + offsets, mask=offsets < WIDTH, other=0).to(tl.float32)
        scale = tl.div_rn(tl.max(tl.abs(x), axis=0), 127.0)
        scale = tl.where(scale == 0, 1.0, scale)
        code = libdevice.rint(tl.div_rn(x, scale)).to(tl.int8)
    else:
        address = old + old_row * (WIDTH + 4)
        scale = tl.load((address + WIDTH).to(tl.pointer_type(tl.float32)))
        code = tl.load(address + offsets, mask=offsets < WIDTH, other=0).to(tl.int8, bitcast=True)
    tl.store(target + offsets, code.to(tl.uint8, bitcast=True), mask=offsets < WIDTH)
    tl.store((target + WIDTH).to(tl.pointer_type(tl.float32)), scale)
    tl.store(restored + new_row * WIDTH + offsets, code.to(tl.float32) * scale, mask=offsets < WIDTH)


@triton.jit
def _append_decode(old_k, old_v, new_k, new_v, pk, pv, k, v, past,
                   kbs, khs, vbs, vhs, HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    position = tl.program_id(1)
    old_row = head * past + position
    new_row = head * (past + 1) + position
    offsets = tl.arange(0, BLOCK)
    kb = (head // HEADS) * kbs + (head % HEADS) * khs
    vb = (head // HEADS) * vbs + (head % HEADS) * vhs
    _append_vector(old_k, new_k, pk, k, old_row, new_row, kb, position == past, offsets, WIDTH)
    _append_vector(old_v, new_v, pv, v, old_row, new_row, vb, position == past, offsets, WIDTH)


def append_cuda(old_k, old_v, new_k, new_v):
    batch, heads, past, stride = old_k.shape
    width = stride - 4
    packed_shape = (batch, heads, past + 1, stride)
    float_shape = (batch, heads, past + 1, width)
    pk, pv = (torch.empty(packed_shape, device=old_k.device, dtype=torch.uint8) for _ in range(2))
    k, v = (torch.empty(float_shape, device=old_k.device, dtype=new_k.dtype) for _ in range(2))
    _append_decode[(batch * heads, past + 1)](
        old_k.contiguous(), old_v.contiguous(), new_k, new_v, pk, pv, k, v, past,
        new_k.stride(0), new_k.stride(1), new_v.stride(0), new_v.stride(1),
        heads, width, triton.next_power_of_2(width),
    )
    return pk, pv, k, v
