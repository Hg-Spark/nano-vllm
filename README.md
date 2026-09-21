# Nano-vLLM — Qwen3.5-MoE Hybrid Runtime

This fork is intentionally specialized for **Qwen3.5-MoE text inference**. The
goal is to keep the runtime small enough to inspect while exposing the three
persistent/conditional execution problems that matter for this model family:

```text
Full Attention -> paged KV cache
Gated DeltaNet -> Conv state + recurrent matrix
Sparse MoE -> dynamic Top-K expert routing
```

## Supported scope

- Qwen3.5-MoE text tower, including `Qwen3.5-35B-A3B`
- BF16/non-quantized safetensors checkpoints
- single GPU, `tensor_parallel_size=1`
- eager execution
- gated Full Attention + partial RoPE
- Gated DeltaNet with persistent per-request state
- sparse routed experts + gated shared expert
- variable-length packed prefill across heterogeneous requests
- continuous batching with explicit request-to-GDN-state ownership
- state-aware chunked prefill across scheduler steps with incremental KV block growth
- decode-first token budgeting with round-robin decode fairness
- hybrid preemption that invalidates KV + GDN state together
- bounded joint prefix caching with ref-counted KV blocks + BF16 GDN snapshots
- token-by-token decode
- deterministic greedy decoding (`temperature=0`)
- multiple EOS token ids from `generation_config.json`

Deliberately out of scope:

- Qwen3 / dense Qwen3.5
- vision tower and MTP
- multimodal packed prefill and 3-axis mRoPE
- quantized checkpoints
- tensor/expert parallelism
- CUDA Graph
- fused GDN or fused/grouped MoE kernels

The current GDN and MoE implementations are **correctness references**, not
performance targets.

## Why specialize the runtime?

A general inference engine must abstract over many model families, cache
backends, parallel modes and kernels. This project has a narrower purpose:
make the Qwen3.5-MoE execution invariants visible before optimizing them.

The runtime therefore keeps only abstractions that correspond to different
ownership/lifetime rules:

- `BlockManager`: paged KV blocks that grow with sequence length;
- `StateSlotManager`: fixed-size recurrent state per active request;
- `Scheduler`: admission, decode priority, chunked prefill and preemption;
- `ModelRunner`: GPU execution and persistent-memory accounting.

See:

- `docs/qwen35_moe_eager_design.md` — current implementation and invariants;
- `docs/qwen35_moe_decisions.md` — design choices, rejected abstractions and
  lessons taken from Transformers/vLLM/SGLang;
- `docs/qwen35_batching_state_design.md` — variable-length prefill, continuous
  batching, recurrent-state ownership and chunk continuity;
- `docs/qwen35_scheduler_preemption_prefix_design.md` — decode-first scheduling,
  hybrid preemption, joint KV/GDN prefix reuse and interview reasoning.

## Installation

```bash
pip install -e .
```

The reference path expects a CUDA environment with FlashAttention available.

## Model

Example:

```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B \
  --local-dir ~/huggingface/Qwen3.5-35B-A3B
```

Set the path used by examples:

```bash
export NANOVLLM_MODEL=~/huggingface/Qwen3.5-35B-A3B
```

## Quick start

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3.5-35B-A3B",
    max_num_batched_tokens=256,
    max_num_seqs=2,
    max_num_state_slots=2,
)

outputs = llm.generate(
    ["Explain sparse MoE inference."],
    SamplingParams(temperature=0.0, max_tokens=64),
)

print(outputs[0]["text"])
llm.exit()
```

## Validation

CPU-level invariants:

```bash
pytest tests
```

Real-checkpoint greedy parity plus BF16 prefix-resume parity:

```bash
python verify_qwen3_5_moe.py /path/to/Qwen3.5-MoE \\
  --check-prefix-resume
```

If token parity fails, locate the first decoder layer with material numerical
divergence:

```bash
python diagnose_qwen3_5_moe.py /path/to/Qwen3.5-MoE
```

Serving benchmark with TTFT, TPOT, P50/P99 latency, prefill/decode execution
throughput and peak CUDA memory:

```bash
python bench.py /path/to/Qwen3.5-MoE \\
  --num-requests 8 \\
  --min-input-len 64 \\
  --max-input-len 256 \\
  --output-len 32 \\
  --json
```

The real-checkpoint comparison is the final correctness gate. Unit tests prove
lifecycle/routing invariants, the prefix-resume probe exercises the actual BF16
GDN checkpoint path, and the layer probe narrows any HF mismatch before kernel
optimization begins.

## Optimization roadmap

```text
real-checkpoint numerical parity
        |
profile eager GDN + eager MoE
        |
fused/chunked GDN kernels
        |
grouped/fused MoE dispatch
        |
capture-safe runtime / CUDA Graph
        |
optional TP/EP only when a concrete target requires it
```
