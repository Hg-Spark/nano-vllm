from __future__ import annotations

import argparse
import gc
import math
import traceback
from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn

from scripts._common import (
    ValidationFailure,
    add_dtype_argument,
    add_hf_experts_backend_argument,
    add_model_argument,
    add_result_argument,
    execute,
    normalize_paths,
    validate_finite,
    validate_model_family,
    validate_positive,
)


def _find_hf_layers(model) -> nn.ModuleList:
    from torch import nn

    for path in (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    ):
        try:
            layers = attrgetter(path)(model)
            if isinstance(layers, nn.ModuleList):
                return layers
        except AttributeError:
            continue
    raise RuntimeError("could not locate Qwen3.5 decoder layers in the HF model")


def _extract_hidden(output) -> torch.Tensor:
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"decoder layer hook returned unsupported output type: {type(output)!r}")


def _capture_hooks(layers, sink):
    if not layers or len(sink) != len(layers):
        raise RuntimeError("layer capture must preallocate every expected decoder layer")
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            if sink[index] is not None:
                raise RuntimeError(f"decoder layer {index} was captured more than once")
            sink[index] = _extract_hidden(output).detach().float().cpu()

        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


def _complete_capture(captured, expected, label):
    if expected <= 0 or len(captured) != expected or any(value is None for value in captured):
        raise RuntimeError(f"{label} layer probe missed one or more decoder layers")
    return captured


def capture_hf_layers(model_path, input_ids, dtype: str, hf_experts_backend: str | None = "eager"):
    import torch
    from transformers import AutoModelForCausalLM

    model = layers = ids = None
    handles = []
    try:
        options = (
            {"experts_implementation": hf_experts_backend} if hf_experts_backend is not None else {}
        )
        model = (
            AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), **options)
            .cuda()
            .eval()
        )
        layers = _find_hf_layers(model)
        captured = [None] * len(layers)
        handles = _capture_hooks(layers, captured)
        ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, return_dict=True)
        return _complete_capture(captured, len(layers), "HF")
    except BaseException as exc:
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        ids = layers = model = None
        gc.collect()
        torch.cuda.empty_cache()


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
    handles = []
    try:
        layers = llm.model_runner.model.model.layers
        captured = [None] * len(layers)
        handles = _capture_hooks(layers, captured)
        llm.generate(
            [input_ids],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        return _complete_capture(captured, len(layers), "nano-vLLM")
    except BaseException as exc:
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        for handle in handles:
            handle.remove()
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
    validate_positive(parser, args, "max_probe_tokens", "max_model_len")
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
