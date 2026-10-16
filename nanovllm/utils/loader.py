import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def _map_weight_name(
    model: nn.Module | type[nn.Module],
    weight_name: str,
) -> str | None:
    skip_prefixes = getattr(
        model,
        "checkpoint_skip_prefixes",
        (),
    )
    for prefix in skip_prefixes:
        if weight_name.startswith(prefix):
            return None

    prefix_map = getattr(
        model,
        "checkpoint_prefix_map",
        (),
    )
    for source_prefix, target_prefix in prefix_map:
        if weight_name.startswith(source_prefix):
            return target_prefix + weight_name[len(source_prefix) :]
    return weight_name


def load_model(model: nn.Module, path: str) -> None:
    expected = dict(model.named_parameters())
    canonical_by_id = {id(param): name for name, param in expected.items()}
    aliases = {
        name: canonical_by_id[id(param)]
        for name, param in model.named_parameters(remove_duplicate=False)
    }
    loaded: set[str] = set()
    packed_shards = getattr(model, "packed_checkpoint_shards", lambda: {})()
    required_shards: dict[str, set[str]] = {}
    for name, (target, _, _) in packed_shards.items():
        required_shards.setdefault(target, set()).add(name)
    seen_shards: set[str] = set()
    checkpoint_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not checkpoint_files:
        raise FileNotFoundError(f"no safetensors checkpoint files found under {path}")

    for file in checkpoint_files:
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                weight_name = _map_weight_name(
                    model,
                    checkpoint_name,
                )
                if weight_name is None:
                    continue

                shard = packed_shards.get(weight_name)
                canonical_name = shard[0] if shard is not None else aliases.get(weight_name)
                param = expected.get(canonical_name)
                if param is None:
                    raise KeyError(
                        f"unexpected checkpoint weight: {checkpoint_name!r} -> {weight_name!r}"
                    )

                loaded_weight = f.get_tensor(checkpoint_name)
                transform = getattr(
                    model,
                    "transform_checkpoint_weight",
                    None,
                )
                if transform is not None:
                    loaded_weight = transform(
                        weight_name,
                        loaded_weight,
                    )
                destination = param if shard is None else param[shard[1] : shard[2]]
                if tuple(destination.shape) != tuple(loaded_weight.shape):
                    raise ValueError(
                        "checkpoint shape mismatch for "
                        f"{checkpoint_name!r}: checkpoint="
                        f"{tuple(loaded_weight.shape)}, model="
                        f"{tuple(destination.shape)}"
                    )
                if shard is not None:
                    if weight_name in seen_shards or canonical_name in loaded:
                        raise ValueError(f"duplicate packed checkpoint shard: {weight_name}")
                    destination.data.copy_(loaded_weight)
                    seen_shards.add(weight_name)
                    if required_shards[canonical_name].issubset(seen_shards):
                        loaded.add(canonical_name)
                elif canonical_name in loaded:
                    if not torch.equal(param.detach(), loaded_weight.to(param)):
                        raise ValueError(
                            "conflicting checkpoint weights for shared parameter: "
                            f"{checkpoint_name!r} -> {canonical_name!r}"
                        )
                else:
                    param.data.copy_(loaded_weight)
                    loaded.add(canonical_name)

    missing = sorted(set(expected) - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise KeyError(f"missing model weights after loading: {preview}{suffix}")
