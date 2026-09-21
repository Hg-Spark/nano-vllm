# Qwen3.5-MoE Validation and Benchmark Closure

This document defines the completion gate for the text-only Qwen3.5-MoE hybrid
runtime. It deliberately does not add multimodal, TP/EP, CUDA Graph or fused
kernels to the stage 1-9 target.

## 1. Correctness order

Run the checks in this order:

```text
CPU/runtime invariants
        |
HF greedy token parity
        |
BF16 joint-prefix resume parity
        |
layer-wise probe only if parity fails
        |
serving benchmark
```

### 1.1 Unit tests

```bash
pytest tests
```

Important coverage includes:

- packed variable-length prefill metadata;
- request-to-GDN-state-slot ownership;
- state continuity across chunks and decode;
- decode-first scheduling and fairness;
- atomic KV + GDN preemption recovery;
- joint prefix boundary validation;
- BF16 GDN snapshot storage and continuation semantics.

The BF16 continuation unit test compares resumed execution with an explicit
BF16-quantized reference state. This proves that restore semantics are exact for
the chosen checkpoint representation; it does not replace a real-checkpoint
generation test.

### 1.2 Real checkpoint parity

```bash
python verify_qwen3_5_moe.py /path/to/Qwen3.5-MoE \
  --max-new-tokens 16 \
  --check-prefix-resume
```

The first gate compares deterministic greedy completion tokens against
Transformers.

The second gate constructs two prompts with one full KV block of shared tokens.
It first measures the second prompt with prefix caching disabled, then publishes
a joint KV/GDN prefix from the first prompt and executes the second prompt from
that restored boundary. The greedy completion must remain identical.

The probe also reports the BF16 host snapshot size for the reused boundary.

## 2. Layer-wise diagnosis

Only run this when HF token parity fails:

```bash
python diagnose_qwen3_5_moe.py /path/to/Qwen3.5-MoE \
  --max-probe-tokens 64
```

The script loads the Transformers model first, captures each decoder-layer
output to CPU, releases that model, then runs the same token ids through
nano-vLLM. Keeping only one model resident at a time avoids turning the
diagnostic into an artificial two-model GPU-memory requirement.

For every decoder layer it reports:

```text
max_abs
mean_abs
rms_error
relative_rms_error
```

The first layer above the configured relative-RMS threshold is a localization
hint, not an automatic proof that the layer itself contains the bug. A mismatch
can originate from its input, checkpoint mapping, positional state or an earlier
state mutation.

A practical drill-down order is:

```text
embedding / position ids
        |
Full Attention or GDN mixer
        |
residual + RMSNorm
        |
MoE router / expert output
        |
next decoder layer
```

## 3. Serving benchmark

```bash
python bench.py /path/to/Qwen3.5-MoE \
  --num-requests 8 \
  --min-input-len 64 \
  --max-input-len 256 \
  --output-len 32 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --json
```

The benchmark submits all requests together, so TTFT includes queueing behind
other work in the same offline serving batch.

Reported metrics:

| Metric | Definition |
| --- | --- |
| Request throughput | completed requests / wall-clock time |
| Output throughput | generated tokens / wall-clock time |
| Total token throughput | prompt + generated tokens / wall-clock time |
| TTFT | submission to first generated token |
| TPOT | time from first to final generated token divided by remaining output tokens |
| E2E latency | submission to request completion |
| Prefill execution throughput | scheduled prefill tokens / synchronized prefill GPU time |
| Decode execution throughput | scheduled decode tokens / synchronized decode GPU time |
| Peak CUDA memory | peak allocated CUDA bytes after benchmark reset |

TTFT, TPOT and E2E report mean, P50 and P99.

The benchmark synchronizes CUDA around prefill and decode phase timing. This is
intentional: host-side timestamps around asynchronously launched kernels would
under-report execution time.

## 4. Completion criterion

Stages 1-9 are considered closed when all of the following hold on the selected
real checkpoint and target GPU:

1. `pytest tests` passes;
2. Transformers and nano-vLLM greedy tokens match for the chosen parity prompts;
3. BF16 joint-prefix resume preserves the uncached greedy completion;
4. any parity failure can be localized with the layer probe before changing the
   runtime architecture;
5. benchmark output is recorded with hardware, checkpoint, request-shape and
   scheduler configuration.

Fused GDN, grouped/fused MoE, CUDA Graph and TP/EP start a new optimization
phase. They are not prerequisites for closing the current inference-runtime
design.
