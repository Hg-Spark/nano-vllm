import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _map_weight_name(model: nn.Module, weight_name: str) -> str | None:
    """Map a checkpoint key onto the text-only nano-vLLM module tree."""
    for prefix in getattr(model, "skip_weight_prefixes", ()):
        if weight_name.startswith(prefix):
            return None

    for source_prefix, target_prefix in getattr(
        model,
        "weight_name_prefixes",
        (),
    ):
        if weight_name.startswith(source_prefix):
            return target_prefix + weight_name[len(source_prefix):]

    return weight_name


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                weight_name = _map_weight_name(model, checkpoint_name)
                if weight_name is None:
                    continue

                for packed_name, (target_name, shard_id) in (
                    packed_modules_mapping.items()
                ):
                    if packed_name not in weight_name:
                        continue

                    param_name = weight_name.replace(
                        packed_name,
                        target_name,
                    )
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError as exc:
                        raise KeyError(
                            "checkpoint weight does not map to a model "
                            f"parameter: {checkpoint_name!r} -> "
                            f"{param_name!r}"
                        ) from exc

                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(
                        param,
                        f.get_tensor(checkpoint_name),
                        shard_id,
                    )
                    break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError as exc:
                        raise KeyError(
                            "unexpected checkpoint weight after mapping: "
                            f"{checkpoint_name!r} -> {weight_name!r}"
                        ) from exc

                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(
                        param,
                        f.get_tensor(checkpoint_name),
                    )
