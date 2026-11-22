"""Open the AtlaHu compact checkpoint without loading any BF16 master shards."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from compact import PackedEmbedding, extra_mlp_graph, packed_cache
from atlas_runtime.loop_layout import loop_plans


def assign_tensor(model, name, tensor):
    path, _, key = name.rpartition(".")
    module = model.get_submodule(path)
    if key in module._parameters:
        module._parameters[key] = torch.nn.Parameter(tensor, requires_grad=False)
    elif key in module._buffers:
        module._buffers[key] = tensor
    else:
        raise ValueError(f"unknown checkpoint tensor {name}")


@contextmanager
def open_compact(directory, device="cuda"):
    """Open the saved inference tensors and mount this variant's ordinary forward."""
    from atlas_runtime.loop_kernel import certified_binding
    from atlas_runtime.loop_quant_kernel import bound_forward, implementation_identity
    from projection_kernels import fast_prefill, selected_gemv
    from joint_projections import joint_projections

    directory = Path(directory)
    manifest = json.loads((directory / "atlas-compact.json").read_bytes())
    if manifest["linear_implementation_sha256"] != implementation_identity():
        raise ValueError("installed quantized linear kernel differs from the saved deployment")
    options = manifest["options"]
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)
        if options["embedding_group"]:
            rows, width = manifest["embedding_shape"]
            group = options["embedding_group"]
            model.set_input_embeddings(PackedEmbedding(
                torch.empty((rows, width), dtype=torch.int8), torch.empty((rows, width // group), dtype=torch.float32),
                group, torch.bfloat16,
            ))
    region_table = None
    if "region_scale_dictionary" in manifest:
        with safe_open(str(directory / manifest["region_scale_dictionary"]), framework="pt", device="cpu") as file:
            region_table = file.get_tensor("scale").to(device)
        model.register_buffer("_atlas_region_scale_table", region_table)
    with safe_open(str(directory / "linear-int8.safetensors"), framework="pt", device="cpu") as f:
        for item in manifest["linear"]:
            module = model.get_submodule(item["path"])
            module.weight = None
            module.register_buffer("_atlas_wq", f.get_tensor(item["q"]).to(device))
            module.register_buffer("_atlas_scale", f.get_tensor(item["s"]).to(device))
            module._atlas_group = item["group"]
            if "scale_index_bits" in item:
                from region_codec import indexed_forward
                if region_table is None:
                    raise ValueError("indexed regional weights require their scale dictionary")
                module.register_buffer("_region_table", region_table, persistent=False)
                module._region_bits = item["scale_index_bits"]
                module.forward = indexed_forward(module)
            else:
                module.forward = bound_forward(module)
    with safe_open(str(directory / "remaining.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            assign_tensor(model, name, f.get_tensor(name).to(device))
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("compact checkpoint left an unmaterialized tensor")
    model.eval()
    ops = {}
    loaded = SimpleNamespace(model=model, meta={"device": device, "loop_execution": {"operators": ops}})
    certificates = json.loads((directory / "execution.json").read_bytes())["certificates"]
    with ExitStack() as scope:
        if options["fast_prefill"]:
            scope.enter_context(fast_prefill(model))
        if options.get("gemv_tiles"):
            scope.enter_context(selected_gemv(model, options["gemv_tiles"]))
        for certificate in certificates:
            name = certificate["recipe"]["implementation"]
            # Provider helper replacement inspects the original linear-attention forward.
            # Install sibling projections after those replacements and before graph recording.
            if name == "cuda_graph_decode" and options.get("joint_projections"):
                loaded.meta["joint_projections"] = scope.enter_context(joint_projections(model, options["joint_projections"]))
            if name == "cuda_graph_decode" and options.get("region_gated_norms"):
                from region_codec import bind_gated_norms
                scope.enter_context(bind_gated_norms(model, options["region_gated_norms"],
                                                    options.get("region_norm_arithmetic", "fused")))
            receipt = scope.enter_context(certified_binding(loaded, device, certificate, ops))
            ops[name] = {"status": "active", "calls": receipt["calls"], "bound_modules": receipt["bound_modules"]}
        plans = loop_plans(model)
        if options["extra_mlp_graph"]:
            scope.enter_context(extra_mlp_graph(loaded, plans))
        if options["int8_kv"]:
            scope.enter_context(packed_cache(plans.text, options.get("fused_kv_append", False)))
        yield loaded
