import warnings

import torch
from flashinfer.fused_moe import cutlass_fused_moe_workspace_size


def initialize_moe_runtime(model, config):
    """Allocate the shared FlashInfer fused-MoE decode workspace."""
    if model is None:
        return None, None

    model_dtype = next(model.parameters()).dtype
    if model_dtype not in (torch.float16, torch.bfloat16):
        return None, None

    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if arch not in {89, 90, 100, 103, 107, 110, 120, 121}:
        return None, None

    text_config = config.text_config
    try:
        workspace_bytes = cutlass_fused_moe_workspace_size(
            config.max_num_seqs,
            text_config.hidden_size,
            text_config.moe_intermediate_size,
            text_config.num_experts,
            text_config.num_experts_per_tok,
            x_dtype=model_dtype,
            weight_dtype=model_dtype,
            output_dtype=model_dtype,
            use_fused_finalize=False,
            device=torch.device("cuda"),
        )
        workspace = torch.empty(
            workspace_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        output = torch.empty(
            config.max_num_seqs,
            text_config.hidden_size,
            dtype=model_dtype,
            device="cuda",
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        warnings.warn(
            "FlashInfer fused MoE workspace allocation ran out of memory; "
            f"using reference decode path: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None
    except (NotImplementedError, RuntimeError) as exc:
        warnings.warn(
            "FlashInfer fused MoE is unavailable on this runtime; "
            f"using reference decode path: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None

    model.bind_moe_runtime(workspace, output)
    return workspace, output
