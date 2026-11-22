"""One-token gated delta recurrence. The split implementation retains Torch's reduction order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted(
    state,
    key,
    decay,
    weighted,
    elements: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    block: tl.constexpr,
):
    ix = tl.program_id(0) * block + tl.arange(0, block)
    mask = ix < elements
    h = ix // (K * V)
    s = tl.load(state + ix, mask, other=0.0).to(tl.float32)
    d = tl.load(decay + h, mask, other=0.0).to(tl.float32)
    k = tl.load(key + ix // V, mask, other=0.0).to(tl.float32)
    tl.store(weighted + ix, (s * d) * k, mask)


@triton.jit
def _state_output(
    state,
    key,
    query,
    value,
    beta,
    decay,
    memory,
    new_state,
    weighted,
    elements: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    block: tl.constexpr,
):
    ix = tl.program_id(0) * block + tl.arange(0, block)
    mask = ix < elements
    h = ix // (K * V)
    vix = h * V + ix % V
    s = tl.load(state + ix, mask, other=0.0).to(tl.float32)
    d = tl.load(decay + h, mask, other=0.0).to(tl.float32)
    b = tl.load(beta + h, mask, other=0.0).to(tl.float32)
    v = tl.load(value + vix, mask, other=0.0).to(tl.float32)
    m = tl.load(memory + vix, mask, other=0.0).to(tl.float32)
    k = tl.load(key + ix // V, mask, other=0.0).to(tl.float32)
    q = tl.load(query + ix // V, mask, other=0.0).to(tl.float32)
    result = s * d + k * ((v - m) * b)
    tl.store(new_state + ix, result, mask)
    tl.store(weighted + ix, result * q, mask)


@triton.jit
def _one_pass(
    state,
    key,
    query,
    value,
    beta,
    decay,
    new_state,
    out,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    h = tl.program_id(0)
    v = tl.program_id(1) * BV + tl.arange(0, BV)
    k = tl.arange(0, BK)
    off = (h * K + k[:, None]) * V + v[None, :]
    mask = (k[:, None] < K) & (v[None, :] < V)
    s = tl.load(state + off, mask, other=0.0).to(tl.float32)
    d = tl.load(decay + h).to(tl.float32)
    b = tl.load(beta + h).to(tl.float32)
    kk = tl.load(key + h * K + k, k < K, other=0.0).to(tl.float32)
    qq = tl.load(query + h * K + k, k < K, other=0.0).to(tl.float32)
    target = tl.load(value + h * V + v, v < V, other=0.0).to(tl.float32)
    decayed = s * d
    memory = tl.sum(decayed * kk[:, None], axis=0)
    result = decayed + kk[:, None] * ((target - memory) * b)[None, :]
    tl.store(new_state + off, result, mask)
    tl.store(out + h * V + v, tl.sum(result * qq[:, None], axis=0), v < V)


def single_step(provider, mode, query, key, value, g, beta, initial_state, normalize):
    b, _, h, k = key.shape
    v = value.shape[-1]
    # At sequence length one, [B,T,H,K] and [B,H,T,K] have the same logical rows.
    q = query.reshape(-1, k).float().contiguous()
    key = key.reshape(-1, k).float().contiguous()
    if normalize:
        if mode == "split":
            q = provider.l2norm(q, dim=-1, eps=1e-6)
            key = provider.l2norm(key, dim=-1, eps=1e-6)
        else:
            q, key = provider.l2norm(torch.stack((q, key)), dim=-1, eps=1e-6).unbind()
    q = q / (k**0.5)
    value = value.contiguous()
    decay = g.float().exp().contiguous()
    beta = beta.contiguous()
    state = initial_state.contiguous()
    final = torch.empty((b, h, k, v), device=query.device, dtype=torch.float32)
    with torch.cuda.device(query.device):
        if mode == "fused":
            out = torch.empty_like(value)
            _one_pass[(b * h, triton.cdiv(v, 8))](
                state,
                key,
                q,
                value,
                beta,
                decay,
                final,
                out,
                k,
                v,
                triton.next_power_of_2(k),
                8,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            work = torch.empty_like(final)
            count = final.numel()
            grid = (triton.cdiv(count, 256),)
            _weighted[grid](state, key, decay, work, count, k, v, 256, enable_fp_fusion=False)
            memory = work.sum(-2)
            _state_output[grid](
                state,
                key,
                q,
                value,
                beta,
                decay,
                memory,
                final,
                work,
                count,
                k,
                v,
                256,
                enable_fp_fusion=False,
            )
            out = work.sum(-2).to(query.dtype).reshape(b, 1, h, v)
    return out, final
