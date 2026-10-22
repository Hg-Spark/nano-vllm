"""Teacher-forced, step-by-step Qwen3.5 HF/Nano parity diagnostics.

Run with:
  python -m scripts.trace_qwen3_5 /path/to/model \
      --max-new-tokens 8 --result-file trace.json

The HF greedy token at every step is *forced* into Nano's sampler so that
both engines read identical prefixes after the first prediction mismatch.
No production model/kernel code is changed.
"""

import argparse

from scripts._common import (
    ValidationFailure,
    add_hf_experts_backend_argument,
    add_model_argument,
    add_kv_cache_arguments,
    kv_cache_kwargs,
    add_result_argument,
    execute,
    normalize_paths,
    run_hf_reference,
    validate_finite,
    validate_positive,
)


def _snapshot_last(value):
    """Snapshot the last token before fused residual operations can mutate it."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a Tensor in a trace hook, got {type(value)!r}")
    return value.reshape(-1, value.shape[-1])[-1].detach().float().cpu().clone()


def _metric(ref, actual):
    import torch

    if ref.shape != actual.shape:
        return {
            "reference_shape": list(ref.shape),
            "actual_shape": list(actual.shape),
            "matched_shape": False,
        }
    if not torch.isfinite(ref).all().item() or not torch.isfinite(actual).all().item():
        return {"matched_shape": True, "finite": False}
    delta = actual.float() - ref.float()
    return {
        "matched_shape": True,
        "finite": True,
        "max_abs": delta.abs().max().item(),
        "relative_rmse": (
            delta.square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-12)
        ).item(),
    }


def _topk(logits, count):
    values, ids = logits.topk(min(count, logits.numel()))
    return [{"token": int(i), "logit": float(v)} for i, v in zip(ids.tolist(), values.tolist())]


def _text_backbone(model):
    text = getattr(model, "model", None)
    if text is None:
        raise RuntimeError("unable to find decoder backbone")
    return getattr(text, "language_model", text)


class _LayerCapture:
    def __init__(self, text_model, *, nano):
        self.current = {}
        self.handles = []
        self.nano = nano

        def embed_hook(_module, _inputs, output):
            self.current = {"embedding": _snapshot_last(output)}

        def layer_hook(index):
            def capture(_module, _inputs, output):
                if self.nano:
                    # Nano uses fused additive residuals. The decoder layer returns
                    # (MLP output, accumulated attention residual); their sum is
                    # the value returned by the equivalent HF decoder layer.
                    hidden, residual = output
                    value = hidden.detach().float() + residual.detach().float()
                else:
                    value = output
                self.current[f"layer.{index}"] = _snapshot_last(value)

            return capture

        def final_hook(_module, _inputs, output):
            self.current["norm"] = _snapshot_last(output)

        self.handles.append(text_model.embed_tokens.register_forward_hook(embed_hook))
        for index, layer in enumerate(text_model.layers):
            self.handles.append(layer.register_forward_hook(layer_hook(index)))
        self.handles.append(text_model.norm.register_forward_hook(final_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def snapshot(self):
        expected = len(self.handles)
        if len(self.current) != expected:
            raise RuntimeError(
                f"layer trace incomplete: captured {len(self.current)} of {expected} sites"
            )
        return dict(self.current)


def _hf_states(cache, text_model):
    states = {}
    if cache is None:
        raise RuntimeError("HF forward did not return a cache")
    for index, layer in enumerate(text_model.layers):
        if getattr(layer, "block_type", None) != "linear_attention":
            continue
        conv = cache.get_conv_state(index, state_idx=0)
        recurrent = cache.get_recurrent_state(index, state_idx=0)
        if conv is None or recurrent is None:
            raise RuntimeError(f"HF GDN state missing at layer {index}")
        # HF: [batch, channels, history] and [batch, heads, K, V].
        # Nano: [channels, history] and [heads, V, K].
        states[f"gdn.{index}.conv"] = conv[0].detach().float().cpu().clone()
        states[f"gdn.{index}.recurrent"] = (
            recurrent[0].transpose(-1, -2).detach().float().cpu().clone()
        )
    return states


def _nano_states(text_model, slot):
    states = {}
    for index, layer in enumerate(text_model.layers):
        if layer.block_type != "linear_attention":
            continue
        conv, recurrent = layer.linear_attn.state_pool.slot(slot)
        states[f"gdn.{index}.conv"] = conv.detach().float().cpu().clone()
        states[f"gdn.{index}.recurrent"] = recurrent.detach().float().cpu().clone()
    return states


def trace_hf(model_path, prompt_ids, steps, experts_backend, trace_states):
    import torch
    from transformers import AutoModelForCausalLM

    def evaluate(model):
        backbone = _text_backbone(model)
        capture = _LayerCapture(backbone, nano=False)
        generated, traces = [], []
        past = None
        device = next(model.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        try:
            with torch.inference_mode():
                for step in range(steps):
                    result = model(
                        input_ids=ids,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = result.past_key_values
                    logits = _snapshot_last(result.logits)
                    predicted = int(logits.argmax().item())
                    layers = capture.snapshot()
                    states = _hf_states(past, backbone) if trace_states else {}
                    traces.append({"logits": logits, "layers": layers, "states": states})
                    generated.append(predicted)
                    ids = torch.tensor([[predicted]], dtype=torch.long, device=device)
        finally:
            capture.close()
        return generated, traces

    return run_hf_reference(AutoModelForCausalLM, model_path, experts_backend, evaluate)


def trace_nano(model_path, prompt_ids, forced_tokens, max_model_len, trace_states, kv_options):
    from nanovllm import LLM, SamplingParams

    # A single full prompt prefill, no prefix reuse, and eager decode ensure
    # one unambiguous logits/hidden-state capture for each generated token.
    llm = LLM(
        model_path,
        max_model_len=max_model_len,
        **kv_options,
        max_num_seqs=1,
        max_num_batched_tokens=max(len(prompt_ids), 16),
        max_prefix_cache_entries=0,
        decode_graph_batch_sizes=(),
        max_visual_tokens_per_request=0,
    )
    runner = llm.model_runner
    capture = _LayerCapture(runner.model.model, nano=True)
    original_sampler = runner._sample_logits
    traces = []
    try:

        def forced_sampler(logits, seqs, sample_indices):
            if len(sample_indices) != 1 or len(seqs) != 1:
                raise RuntimeError("teacher forcing requires one sampled request")
            step = len(traces)
            if step >= len(forced_tokens):
                raise RuntimeError("Nano generated more steps than the HF trace")
            layers = capture.snapshot()
            slot = seqs[sample_indices[0]].state_slot
            states = _nano_states(runner.model.model, slot) if trace_states else {}
            traces.append(
                {
                    "logits": _snapshot_last(logits),
                    "layers": layers,
                    "states": states,
                }
            )
            return [forced_tokens[step]]

        runner._sample_logits = forced_sampler
        output = llm.generate(
            [prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=len(forced_tokens), ignore_eos=True),
            use_tqdm=False,
        )
        actual = output[0]["token_ids"]
        if actual != forced_tokens or len(traces) != len(forced_tokens):
            raise RuntimeError(
                "teacher forcing did not feed every HF token: "
                f"expected={len(forced_tokens)}, got={len(traces)}, output={actual}"
            )
        return traces
    finally:
        runner._sample_logits = original_sampler
        capture.close()
        llm.exit()


def compare_traces(hf_tokens, hf_traces, nano_traces, threshold, top_k):
    per_step = []
    first_token_mismatch = None
    first_large_error = None

    for step, (hf, nano) in enumerate(zip(hf_traces, nano_traces)):
        hlog, nlog = hf["logits"], nano["logits"]
        logit_metric = _metric(hlog, nlog)
        hf_argmax, nano_argmax = int(hlog.argmax()), int(nlog.argmax())
        if hf_argmax != nano_argmax and first_token_mismatch is None:
            first_token_mismatch = step
        keys_hf = list(hf["layers"])
        keys_nano = list(nano["layers"])
        if keys_hf != keys_nano:
            raise RuntimeError(f"layer trace sites differ: HF={keys_hf}, Nano={keys_nano}")
        layer_metrics = {key: _metric(hf["layers"][key], nano["layers"][key]) for key in keys_hf}
        if hf["states"].keys() != nano["states"].keys():
            raise RuntimeError("GDN state trace sites differ between HF and Nano")
        state_metrics = {
            key: _metric(hf["states"][key], nano["states"][key]) for key in hf["states"]
        }

        # Report the earliest layer before the final logits. GDN cache errors
        # are reported separately and may explain divergence at later steps.
        for site, metric in [
            *layer_metrics.items(),
            *state_metrics.items(),
            ("logits", logit_metric),
        ]:
            if metric.get("relative_rmse", float("inf")) > threshold and first_large_error is None:
                first_large_error = {"step": step, "site": site, "metrics": metric}

        per_step.append(
            {
                "step": step,
                "phase": "prefill" if step == 0 else "decode",
                "forced_token": int(hf_tokens[step]),
                "hf_argmax": hf_argmax,
                "nano_argmax": nano_argmax,
                "logits": logit_metric,
                "hf_topk": _topk(hlog, top_k),
                "nano_topk": _topk(nlog, top_k),
                "layers": layer_metrics,
                "gdn_states": state_metrics,
            }
        )
    return {
        "steps": len(per_step),
        "first_token_mismatch": first_token_mismatch,
        "first_error_above_threshold": first_large_error,
        "relative_rmse_threshold": threshold,
        "per_step": per_step,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser, required=True)
    add_kv_cache_arguments(parser)
    add_hf_experts_backend_argument(parser)
    parser.add_argument(
        "--prompt", default="Explain why KV cache is useful in autoregressive inference."
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--relative-rmse-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--trace-gdn-states", action="store_true")
    parser.add_argument("--verbose-layers", action="store_true")
    add_result_argument(parser)
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens", "max_model_len", "top_k")
    validate_finite(
        parser,
        args.relative_rmse_threshold,
        "relative-rmse-threshold",
        minimum=0,
        inclusive=False,
    )
    return normalize_paths(args, "model", "result_file")


def run(args):
    from transformers import AutoConfig, AutoTokenizer

    from nanovllm.config import is_qwen3_5_moe

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt tokenization produced no input IDs")
    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(cfg, "text_config", cfg)
    effective_len = min(args.max_model_len, text_cfg.max_position_embeddings)
    if len(prompt_ids) + args.max_new_tokens - 1 > effective_len:
        raise ValueError("prompt and decode steps exceed effective max_model_len")

    backend = args.hf_experts_backend if is_qwen3_5_moe(cfg) else None
    hf_tokens, hf = trace_hf(
        args.model,
        prompt_ids,
        args.max_new_tokens,
        backend,
        args.trace_gdn_states,
    )
    nano = trace_nano(
        args.model,
        prompt_ids,
        hf_tokens,
        args.max_model_len,
        args.trace_gdn_states,
        kv_cache_kwargs(args),
    )
    result = compare_traces(hf_tokens, hf, nano, args.relative_rmse_threshold, args.top_k)
    result.update(
        {
            "prompt_length": len(prompt_ids),
            "forced_tokens": hf_tokens,
            "hf_experts_backend": backend,
        }
    )
    print("Step  Phase     HF top1  Nano top1  Logits relRMSE  Max abs")
    for row in result["per_step"]:
        metrics = row["logits"]
        print(
            f"{row['step']:>4}  {row['phase']:<8}"
            f"{row['hf_argmax']:>8}  {row['nano_argmax']:>10}"
            f"  {metrics.get('relative_rmse', float('nan')):>14.6g}"
            f"  {metrics.get('max_abs', float('nan')):>8.5g}"
        )
        if args.verbose_layers:
            for site, values in [*row["layers"].items(), *row["gdn_states"].items()]:
                print(f"         {site:<22} {values}")
    print(f"First token mismatch (0-based): {result['first_token_mismatch']}")
    print(f"First site above threshold: {result['first_error_above_threshold']}")

    if result["first_token_mismatch"] is not None or result["first_error_above_threshold"]:
        raise ValidationFailure("FAIL: HF/Nano teacher-forced parity trace differs", result)
    print("PASS: teacher-forced tokens and selected numerical metrics match")
    return result


def main(argv=None):
    return execute("qwen35_teacher_forced_trace", parse_args(argv), run)


if __name__ == "__main__":
    raise SystemExit(main())
