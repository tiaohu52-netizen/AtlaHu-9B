"""CUDA inference specialization of the one-token causal conv update.

The reference implementation concatenates the cached window with the new token,
copies the shifted window back into the cache, evaluates a grouped conv1d over
the concatenated columns and applies the stored activation. This module performs
the same four steps for seq_len == 1 in a single kernel while keeping the
reference dtype path: BF16 operands, FP32 sequential accumulation in tap order,
one rounding to BF16 after the convolution and a second rounding after the
activation. The cache window is updated in place, so the caller's tensor must be
a contiguous view.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_step(
    state_ptr,
    input_ptr,
    weight_ptr,
    output_ptr,
    channels,
    STATE_LEN: tl.constexpr,
    KERNEL: tl.constexpr,
    ACTIVATION: tl.constexpr,
    block: tl.constexpr,
):
    ix = tl.program_id(0) * block + tl.arange(0, block)
    mask = ix < channels
    base = ix * STATE_LEN
    acc = tl.zeros((block,), dtype=tl.float32)
    # The reference concatenates the state with the new token and takes the last
    # KERNEL columns of that concatenation as the receptive field.
    first = STATE_LEN + 1 - KERNEL
    for k in tl.static_range(KERNEL):
        column = first + k
        if column < STATE_LEN:
            value = tl.load(state_ptr + base + column, mask=mask, other=0.0).to(tl.float32)
        else:
            value = tl.load(input_ptr + ix, mask=mask, other=0.0).to(tl.float32)
        tap = tl.load(weight_ptr + ix * KERNEL + k, mask=mask, other=0.0).to(tl.float32)
        acc += tap * value
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    if ACTIVATION == 1:
        rounded = (rounded / (1 + tl.exp(-rounded))).to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + ix, rounded.to(tl.bfloat16), mask=mask)
    # Slide the cache window: drop the oldest column and append the new token.
    for k in tl.static_range(STATE_LEN - 1):
        column = k + 1
        if column < STATE_LEN:
            shifted = tl.load(state_ptr + base + column, mask=mask, other=0.0)
        else:
            shifted = tl.load(input_ptr + ix, mask=mask, other=0.0)
        tl.store(state_ptr + base + k, shifted, mask=mask)
    tail = tl.load(input_ptr + ix, mask=mask, other=0.0)
    tl.store(state_ptr + base + STATE_LEN - 1, tail, mask=mask)


def fused_conv_update(hidden_states, conv_state, weight, bias, activation=None):
    """Replace causal_conv1d_update for single-token cached decode."""
    if bias is not None:
        raise NotImplementedError("needs_backend: fused conv update without bias")
    channels, kernel = weight.shape
    state_len = conv_state.shape[-1]
    if hidden_states.shape[-1] != 1 or hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise NotImplementedError("needs_backend: single-token BF16 cached decode")
    if activation not in (None, "silu"):
        raise NotImplementedError(f"needs_backend: fused activation {activation}")
    flat = hidden_states.reshape(-1)
    state = conv_state.view(conv_state.shape[0] * channels, state_len)
    flat = flat.contiguous()
    out = torch.empty_like(flat)
    with torch.cuda.device(flat.device):
        _conv_step[(triton.cdiv(channels, 256),)](
            state,
            flat,
            weight,
            out,
            channels,
            state_len,
            kernel,
            1 if activation == "silu" else 0,
            256,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out.reshape(hidden_states.shape)


def implementation_identity():
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
