import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    map_weight_name = getattr(model, "map_weight_name", None)

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                weight_name = map_weight_name(checkpoint_name) if map_weight_name else checkpoint_name
                if weight_name is None:
                    continue

                for src_name, (dst_name, shard_id) in packed_modules_mapping.items():
                    if src_name in weight_name:
                        param_name = weight_name.replace(src_name, dst_name)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError as exc:
                            raise KeyError(
                                f"Checkpoint weight {checkpoint_name!r} mapped to missing parameter {param_name!r}"
                            ) from exc
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(checkpoint_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError as exc:
                        raise KeyError(
                            f"Checkpoint weight {checkpoint_name!r} mapped to missing parameter {weight_name!r}"
                        ) from exc
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(checkpoint_name))
