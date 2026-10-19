from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from scripts._common import (
    ValidationFailure,
    add_dtype_argument,
    add_hf_experts_backend_argument,
    add_model_argument,
    add_result_argument,
    execute,
    normalize_paths,
    run_hf_reference,
    validate_finite,
    validate_model_family,
    validate_positive,
)


@contextmanager
def _capture_hooks(layers, sink, *, fused_residual=False):
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            hidden = output[0] + output[1] if fused_residual else output
            sink[index] = hidden.detach().float().cpu()

        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _complete_capture(captured, expected, label):
    if expected <= 0 or len(captured) != expected or any(value is None for value in captured):
        raise RuntimeError(f"{label} layer probe missed one or more decoder layers")
    return captured


def capture_hf_layers(model_path, input_ids, dtype: str, hf_experts_backend: str | None = "eager"):
    import torch
    from transformers import AutoModelForCausalLM

    def probe(model):
        layers = model.model.layers
        captured = [None] * len(layers)
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        with _capture_hooks(layers, captured), torch.inference_mode():
            model(input_ids=ids, use_cache=False, return_dict=True)
        return _complete_capture(captured, len(layers), "HF")

    return run_hf_reference(AutoModelForCausalLM, model_path, dtype, hf_experts_backend, probe)


def capture_nano_layers(model_path, input_ids, max_model_len, dtype: str):
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model_path,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(256, len(input_ids)),
        max_num_seqs=1,
        max_prefix_cache_entries=0,
    )
    try:
        layers = llm.model_runner.model.model.layers
        captured = [None] * len(layers)
        with _capture_hooks(layers, captured, fused_residual=True):
            llm.generate(
                [input_ids],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
        return _complete_capture(captured, len(layers), "nano-vLLM")
    finally:
        llm.exit()


def _flatten_hidden(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise RuntimeError(f"expected decoder hidden states with rank 2/3, got {tensor.shape}")
    return tensor


def compare_layers(
    hf_layers: list[torch.Tensor],
    nano_layers: list[torch.Tensor],
    relative_rms_threshold: float,
    details: list | None = None,
) -> int:
    import torch

    if not math.isfinite(relative_rms_threshold) or relative_rms_threshold <= 0:
        raise ValueError("relative RMS threshold must be finite and positive")
    if (
        not hf_layers
        or len(hf_layers) != len(nano_layers)
        or any(value is None for value in [*hf_layers, *nano_layers])
    ):
        raise RuntimeError("layer comparison requires complete, matching captures")

    first_divergent = -1
    header = "layer  shape                 max_abs      mean_abs     rms_error    rel_rms"
    print(header + "\n" + "-" * len(header))

    for index, (hf, nano) in enumerate(zip(hf_layers, nano_layers)):
        hf, nano = _flatten_hidden(hf), _flatten_hidden(nano)
        if hf.shape != nano.shape:
            if details is not None:
                details.append(
                    {
                        "layer": index,
                        "hf_shape": list(hf.shape),
                        "nano_shape": list(nano.shape),
                        "shape_match": False,
                    }
                )
            if first_divergent < 0:
                first_divergent = index
            continue

        if (
            hf.numel() == 0
            or not torch.isfinite(hf).all().item()
            or not torch.isfinite(nano).all().item()
        ):
            raise ValidationFailure(
                f"DIVERGENCE: empty or non-finite hidden states at layer {index}",
                {"layer": index, "finite_hidden_states": False},
            )
        diff = nano - hf
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        rms_error = diff.square().mean().sqrt().item()
        ref_rms = hf.square().mean().sqrt().item()
        rel_rms = rms_error / max(ref_rms, 1e-12)
        if not all(math.isfinite(val) for val in (max_abs, mean_abs, rms_error, ref_rms, rel_rms)):
            raise ValidationFailure(
                f"DIVERGENCE: non-finite error at layer {index}", {"layer": index}
            )
        if details is not None:
            details.append(
                {
                    "layer": index,
                    "shape": list(hf.shape),
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "rms_error": rms_error,
                    "relative_rms": rel_rms,
                }
            )
        print(
            f"{index:>5}  {str(tuple(hf.shape)):<20} "
            f"{max_abs:>11.4e} {mean_abs:>11.4e} {rms_error:>11.4e} {rel_rms:>9.3e}"
        )
        if first_divergent < 0 and rel_rms > relative_rms_threshold:
            first_divergent = index

    return first_divergent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare decoder-layer hidden states between Transformers and nano-vLLM."
    )
    add_model_argument(parser, required=True)
    add_dtype_argument(parser)
    add_hf_experts_backend_argument(parser)
    parser.add_argument(
        "--prompt", default="Explain why recurrent state must align with the KV prefix."
    )
    parser.add_argument("--max-probe-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--relative-rms-threshold", type=float, default=2e-2)
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_probe_tokens")
    validate_finite(parser, args.relative_rms_threshold, "relative-rms-threshold", inclusive=False)
    return normalize_paths(args, "model", "result_file")


def run(args):
    from transformers import AutoTokenizer

    is_moe = validate_model_family(args.model)
    hf_backend = args.hf_experts_backend if is_moe else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    input_ids = tokenizer.encode(args.prompt)[: args.max_probe_tokens]
    if not input_ids:
        raise RuntimeError("layer probe prompt produced no tokens")
    hf_layers = capture_hf_layers(args.model, input_ids, args.dtype, hf_experts_backend=hf_backend)
    nano_layers = capture_nano_layers(args.model, input_ids, args.max_model_len, args.dtype)

    details = []
    first_divergent = compare_layers(hf_layers, nano_layers, args.relative_rms_threshold, details)
    result = {
        "probe_tokens": input_ids,
        "layer_count": len(hf_layers),
        "hf_experts_backend": hf_backend,
        "first_divergent_layer": None if first_divergent < 0 else first_divergent,
        "layers": details,
    }
    if first_divergent < 0:
        print(f"PASS: no decoder layer exceeded threshold ({args.relative_rms_threshold:g})")
        return result
    raise ValidationFailure(
        f"DIVERGENCE: first decoder layer above relative RMS threshold is layer {first_divergent}",
        result,
    )


def main(argv=None):
    return execute("layer_diagnosis", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
