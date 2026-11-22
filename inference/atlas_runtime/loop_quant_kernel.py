"""AtlaHu quantized linear execution."""

import hashlib
from importlib.util import find_spec
from pathlib import Path

import torch


def implementation_identity():
    source = Path(__file__).with_name("cuda_w8a16.py")
    return hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def bound_forward(module):
    """Instance forward that runs the certified int8 expression."""

    triton_available = find_spec("triton") is not None

    def forward(input):
        q = module._atlas_wq
        group = getattr(module, "_atlas_group", 1)
        if (triton_available and input.is_cuda and input.device == q.device
            and input.dtype == torch.bfloat16 and input.shape == (1, 1, q.shape[1])
            and q.shape[1] % group == 0 and module.bias is None and not torch.is_grad_enabled()
        ):
            from .cuda_w8a16 import fused_gemv

            return fused_gemv(input, q, module._atlas_scale, q.shape[0], group=group)
        if group == 1:
            weight = (q.float() * module._atlas_scale[:, None]).to(torch.bfloat16)
        else:
            weight = (q.float() * module._atlas_scale.repeat_interleave(group, dim=1)).to(torch.bfloat16)
        return torch.nn.functional.linear(input, weight, module.bias)

    return forward
