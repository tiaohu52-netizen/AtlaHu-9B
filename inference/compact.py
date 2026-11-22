"""Optional inference components for the SenseNova Loop fixed-token study."""

from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class PackedEmbedding(torch.nn.Module):
    """Store int8 rows and FP32 group scales. Dequantize only lookup results."""

    def __init__(self, codes, scales, group, output_dtype):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.group = group
        self.output_dtype = output_dtype
        self.num_embeddings, self.embedding_dim = codes.shape

    @classmethod
    @torch.no_grad()
    def from_weight(cls, weight, group, device):
        rows, width = weight.shape
        if group < 1 or width % group:
            raise ValueError("embedding width must be divisible by the quantization group")
        codes = torch.empty((rows, width), dtype=torch.int8, device=device)
        scales = torch.empty((rows, width // group), dtype=torch.float32, device=device)
        for start in range(0, rows, 1024):
            values = weight[start:start + 1024].to(device=device, dtype=torch.float32)
            blocks = values.reshape(values.shape[0], -1, group)
            scale = blocks.abs().amax(-1) / 127
            scale = torch.where(scale == 0, 1.0, scale)
            codes[start:start + len(values)].copy_((blocks / scale.unsqueeze(-1)).round().clamp_(-127, 127).reshape_as(values))
            scales[start:start + len(values)].copy_(scale)
        return cls(codes, scales, group, weight.dtype)

    def forward(self, ids):
        selected = self.codes[ids].reshape(*ids.shape, self.embedding_dim // self.group, self.group)
        vectors = selected.float() * self.scales[ids].unsqueeze(-1)
        return vectors.reshape(*ids.shape, self.embedding_dim).to(self.output_dtype)

    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers())

    @property
    def weight(self):
        """Physical int8 storage, also used by the graph engine for device discovery."""
        return self.codes


def _pack_reference(value):
    values = value.float()
    scale = values.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale == 0, 1.0, scale)
    code = (values / scale).round().clamp_(-127, 127).to(torch.int8).view(torch.uint8)
    return torch.cat((code, scale.contiguous().view(torch.uint8)), dim=-1)


def _unpack_reference(packed, dtype):
    code = packed[..., :-4].view(torch.int8)
    scale = packed[..., -4:].contiguous().view(torch.float32)
    return (code.float() * scale).to(dtype)


def pack_pair(keys, values):
    if keys.is_cuda:
        from kv_kernels import pack_cuda
        return pack_cuda(keys, values)
    return _pack_reference(keys), _pack_reference(values)


def unpack_pair(keys, values, dtype):
    if keys.is_cuda:
        from kv_kernels import unpack_cuda
        return unpack_cuda(keys, values, dtype)
    return _unpack_reference(keys, dtype), _unpack_reference(values, dtype)


class PackedKVLayer(DynamicLayer):
    """Each full-attention head/token keeps a signed byte vector plus a FP32 scale."""

    def __init__(self, fused_append=False):
        super().__init__()
        self.fused_append = fused_append

    def update(self, key_states, value_states, *args, **kwargs):
        if (self.fused_append and self.is_initialized and key_states.is_cuda and key_states.shape[-2] == 1
                and key_states.stride(-1) == value_states.stride(-1) == 1):
            from kv_kernels import append_cuda
            self.keys, self.values, k, v = append_cuda(self.keys, self.values, key_states, value_states)
            return k, v
        keys, values = pack_pair(key_states, value_states)
        if self.is_initialized:
            self.keys = torch.cat((self.keys, keys), dim=-2)
            self.values = torch.cat((self.values, values), dim=-2)
        else:
            self.keys, self.values = keys, values
            self.is_initialized = True
            self.device, self.dtype = keys.device, keys.dtype
        return unpack_pair(self.keys, self.values, key_states.dtype)


@contextmanager
def packed_cache(text, fused_append=False):
    """Supply a fresh mixed cache to the ordinary forward; keep recurrent slots native."""
    def before(module, args, kwargs):
        past = kwargs.get("past_key_values")
        if kwargs.get("use_cache") is False or past is not None:
            return
        cache = DynamicCache(config=module._loop_cache_config)
        for i, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer:
                cache.layers[i] = PackedKVLayer(fused_append)
        cache._loop_execution_plan = module._loop_plan
        return args, dict(kwargs, past_key_values=cache)

    hook = text.register_forward_pre_hook(before, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()


@contextmanager
def extra_mlp_graph(loaded, plans):
    """Capture MLPs outside the existing independent linear/residual regions."""
    from atlas_runtime.loop_cuda_graph import GraphExecution
    inherited = loaded.graph_execution
    engine = GraphExecution(loaded, plans, inherited.warmups)
    selected = [name for name, (family, _) in engine.layout.items() if family == "mlp"
                and not any(name == path or name.startswith(path + ".") for path in inherited.regions)]
    engine.active = {"mlp"}
    with engine.mount(selected):
        yield engine
