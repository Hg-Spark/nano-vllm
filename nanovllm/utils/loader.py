import os
from glob import glob

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
            return (
                target_prefix
                + weight_name[len(source_prefix):]
            )
    return weight_name


def load_model(model: nn.Module, path: str):
    expected = dict(model.named_parameters())
    loaded: set[str] = set()
    checkpoint_files = sorted(
        glob(os.path.join(path, "*.safetensors"))
    )
    if not checkpoint_files:
        raise FileNotFoundError(
            f"no safetensors checkpoint files found under {path}"
        )

    for file in checkpoint_files:
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                weight_name = _map_weight_name(
                    model,
                    checkpoint_name,
                )
                if weight_name is None:
                    continue

                param = expected.get(weight_name)
                if param is None:
                    raise KeyError(
                        "unexpected checkpoint weight: "
                        f"{checkpoint_name!r} -> {weight_name!r}"
                    )

                loaded_weight = f.get_tensor(checkpoint_name)
                if tuple(param.shape) != tuple(loaded_weight.shape):
                    raise ValueError(
                        "checkpoint shape mismatch for "
                        f"{checkpoint_name!r}: checkpoint="
                        f"{tuple(loaded_weight.shape)}, model="
                        f"{tuple(param.shape)}"
                    )
                param.data.copy_(loaded_weight)
                loaded.add(weight_name)

    missing = sorted(set(expected) - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise KeyError(
            "missing model weights after loading: "
            f"{preview}{suffix}"
        )
