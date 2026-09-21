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
- token-by-token decode
- deterministic greedy decoding (`temperature=0`)
- multiple EOS token ids from `generation_config.json`

Deliberately out of scope:

- Qwen3 / dense Qwen3.5
- vision tower and MTP
- quantized checkpoints
- tensor/expert parallelism
- cross-request prefix caching
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
  batching, recurrent-state ownership, chunk continuity and interview notes.

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

Real-checkpoint greedy parity:

```bash
python verify_qwen3_5_moe.py /path/to/Qwen3.5-35B-A3B
```

The real-checkpoint comparison is the final correctness gate. Unit tests can
prove lifecycle and routing invariants but cannot prove that every checkpoint
mapping and model equation matches the upstream implementation.

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
