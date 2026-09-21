import argparse
import gc
import os

import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def _getattr_path(obj, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _find_hf_layers(model) -> nn.ModuleList:
    candidates = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
    )
    for path in candidates:
        try:
            layers = _getattr_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList):
            return layers
    raise RuntimeError(
        "could not locate Qwen3.5-MoE decoder layers in the HF model"
    )


def _extract_hidden(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(
        f"decoder layer hook returned unsupported output type: {type(output)!r}"
    )


def _capture_hooks(
    layers,
    sink: list[torch.Tensor | None],
):
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = _extract_hidden(output)
            while len(sink) <= index:
                sink.append(None)
            sink[index] = hidden.detach().float().cpu()
        return hook

    for index, layer in enumerate(layers):
        handles.append(
            layer.register_forward_hook(make_hook(index))
        )
    return handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def capture_hf_layers(
    model_path: str,
    input_ids: list[int],
) -> list[torch.Tensor]:
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).cuda().eval()
    layers = _find_hf_layers(model)
    captured: list[torch.Tensor | None] = []
    handles = _capture_hooks(layers, captured)
    try:
        ids = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device="cuda",
        )
        with torch.inference_mode():
            model(
                input_ids=ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        _remove_hooks(handles)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if any(value is None for value in captured):
        raise RuntimeError("HF layer probe missed one or more decoder layers")
    return [
        value
        for value in captured
        if value is not None
    ]


def capture_nano_layers(
    model_path: str,
    input_ids: list[int],
    max_model_len: int,
) -> list[torch.Tensor]:
    llm = LLM(
        model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(256, len(input_ids)),
        max_num_seqs=1,
        max_num_state_slots=1,
        enable_prefix_cache=False,
    )
    captured: list[torch.Tensor | None] = []
    handles = _capture_hooks(
        llm.model_runner.model.model.layers,
        captured,
    )
    try:
        seq = Sequence(
            input_ids,
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                ignore_eos=True,
            ),
        )
        llm.scheduler.add(seq)
        scheduled = llm.scheduler.schedule()
        if scheduled.decode_seqs or scheduled.prefill_seqs != [seq]:
            raise RuntimeError(
                "layer probe requires one full prefill batch"
            )
        if seq.num_scheduled_tokens != len(input_ids):
            raise RuntimeError(
                "layer probe prompt was chunked unexpectedly: "
                f"scheduled={seq.num_scheduled_tokens}, "
                f"prompt={len(input_ids)}"
            )
        llm.model_runner.run(
            scheduled.prefill_seqs,
            is_prefill=True,
        )
    finally:
        _remove_hooks(handles)
        llm.exit()

    if any(value is None for value in captured):
        raise RuntimeError(
            "nano-vLLM layer probe missed one or more decoder layers"
        )
    return [
        value
        for value in captured
        if value is not None
    ]


def _flatten_hidden(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise RuntimeError(
            f"expected decoder hidden states with rank 2/3, got {tensor.shape}"
        )
    return tensor


def compare_layers(
    hf_layers: list[torch.Tensor],
    nano_layers: list[torch.Tensor],
    relative_rms_threshold: float,
) -> int:
    if len(hf_layers) != len(nano_layers):
        raise RuntimeError(
            "decoder layer count mismatch: "
            f"hf={len(hf_layers)}, nano={len(nano_layers)}"
        )

    first_divergent = -1
    header = (
        "layer  shape                 max_abs      mean_abs     "
        "rms_error    rel_rms"
    )
    print(header)
    print("-" * len(header))

    for index, (hf, nano) in enumerate(zip(hf_layers, nano_layers)):
        hf = _flatten_hidden(hf)
        nano = _flatten_hidden(nano)
        if hf.shape != nano.shape:
            print(
                f"{index:>5}  shape-mismatch "
                f"hf={tuple(hf.shape)} nano={tuple(nano.shape)}"
            )
            if first_divergent < 0:
                first_divergent = index
            continue

        diff = nano - hf
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        rms_error = diff.square().mean().sqrt().item()
        ref_rms = hf.square().mean().sqrt().item()
        rel_rms = rms_error / max(ref_rms, 1e-12)

        print(
            f"{index:>5}  {str(tuple(hf.shape)):<20} "
            f"{max_abs:>11.4e} {mean_abs:>11.4e} "
            f"{rms_error:>11.4e} {rel_rms:>9.3e}"
        )
        if (
            first_divergent < 0
            and rel_rms > relative_rms_threshold
        ):
            first_divergent = index

    return first_divergent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare decoder-layer hidden states between Transformers and "
            "nano-vLLM for one Qwen3.5-MoE text prompt."
        )
    )
    parser.add_argument("model")
    parser.add_argument(
        "--prompt",
        default="Explain why recurrent state must align with the KV prefix.",
    )
    parser.add_argument(
        "--max-probe-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--relative-rms-threshold",
        type=float,
        default=2e-2,
    )
    args = parser.parse_args()

    if args.max_probe_tokens <= 0:
        raise ValueError("max-probe-tokens must be positive")
    if args.relative_rms_threshold <= 0:
        raise ValueError("relative-rms-threshold must be positive")

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    input_ids = tokenizer.encode(args.prompt)
    input_ids = input_ids[:args.max_probe_tokens]
    if not input_ids:
        raise RuntimeError("layer probe prompt produced no tokens")

    print(
        f"Probe tokens: {len(input_ids)}; "
        "capturing HF first so only one model copy is resident at a time."
    )
    hf_layers = capture_hf_layers(
        model_path,
        input_ids,
    )
    nano_layers = capture_nano_layers(
        model_path,
        input_ids,
        args.max_model_len,
    )

    first_divergent = compare_layers(
        hf_layers,
        nano_layers,
        args.relative_rms_threshold,
    )
    if first_divergent < 0:
        print(
            "PASS: no decoder layer exceeded the configured relative RMS "
            f"threshold ({args.relative_rms_threshold:g})"
        )
        return

    raise SystemExit(
        "DIVERGENCE: first decoder layer above relative RMS threshold is "
        f"layer {first_divergent}"
    )


if __name__ == "__main__":
    main()
