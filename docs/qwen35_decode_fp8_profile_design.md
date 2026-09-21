# Decode-first Budget, FP8 KV Cache, and Profile-Gated Decode Optimization

This note records the performance-oriented extension of the Qwen3.5 hybrid
runtime. It is intentionally narrower than a production inference engine. The
goal is to make three engineering decisions explicit and testable:

1. keep decode-first token budgeting while making KV/GDN reservation atomic;
2. add an opt-in FP8 E4M3 paged KV storage path without changing cache
   ownership semantics;
3. profile Full-Attention decode before adding a custom GQA/PagedAttention
   kernel.

The implementation reuses the existing Continuous Batching, State-aware
Chunked Prefill, Hybrid Preemption, and Joint Prefix Cache paths. It does not
introduce another scheduler hierarchy or a generic attention-backend registry.

---

## 1. Why these three changes belong together

For Qwen3.5 hybrid models, the main serving resources are heterogeneous:

```text
Full Attention      -> paged KV blocks
Gated DeltaNet      -> Conv/Recurrent active state slot
Prefix reuse        -> shared KV refs + GDN snapshot
```

Scheduling decides when these resources are reserved. FP8 changes the storage
cost of one resource family. Profiling determines whether the resulting
Full-Attention decode path is important enough to justify a specialized
kernel.

The dependency is therefore:

```text
Decode-first token budget
        |
        v
atomic KV + GDN reservation
        |
        v
FP8 paged KV storage
        |
        v
decode profile
        |
        +-- Full Attention small share --> stop
        |
        +-- Full Attention large share --> benchmark fused GQA/PA kernel
```

This order avoids writing a complex decode kernel before there is evidence that
it affects end-to-end latency.

---

## 2. Decode-first token budget: what changed

The scheduler already used the correct high-level policy:

```text
step budget
  -> running decode requests, one token each
  -> remaining tokens to chunked/new prefill
```

It also already rotated selected decode requests to the back of the running
queue, preventing starvation when:

```text
active decode requests > max_num_batched_tokens
```

The missing piece was failure atomicity during prefill admission.

### 2.1 The failure window

A fresh prefill request needs both:

```text
one GDN state slot
AND
enough KV pages for the scheduled chunk
```

The previous code performed:

```text
allocate state slot
allocate/grow KV pages
```

If KV growth failed after the state slot was acquired, the request could retain
one side of the hybrid reservation.

The scheduler now records the pre-call KV block count, allocates the missing
state slot, grows KV, and rolls back only resources acquired by that call when
an exception occurs.

For an existing chunked-prefill request, committed prefix resources are kept.
Only the speculative KV tail is released.

### 2.2 Why rollback is not a generic transaction object

A production engine may need transaction objects across several cache groups,
host swap, remote KV, speculative branches, and asynchronous transfers.

This runtime has exactly two active history resources:

- paged KV blocks;
- one request-level GDN state slot shared across GDN layers.

A small scheduler helper plus `BlockManager.truncate_blocks()` exposes the
important invariant directly:

```text
reservation failure
=> no new KV ownership survives
AND no newly allocated GDN slot survives
```

A generic resource transaction layer would add more abstraction than behavior.

---

## 3. FP8 KV Cache

### 3.1 Configuration

The runtime now accepts:

```python
LLM(
    model=...,
    kv_cache_dtype="fp8_e4m3",
    kv_cache_k_scale=1.0,
    kv_cache_v_scale=1.0,
)
```

`kv_cache_dtype="auto"` preserves the model activation dtype and remains the
default.

The initial implementation intentionally supports E4M3 only.

### 3.2 Persistent memory accounting

For BF16, one KV element consumes two bytes. For FP8 E4M3, it consumes one.

The paged-cache block budget is calculated with the selected cache element
size:

```text
block_bytes =
    2                       # K and V
  * num_full_attention_layers
  * block_size
  * num_kv_heads
  * head_dim
  * cache_element_size
```

GDN active-state memory is reserved separately before the KV block count is
computed.

This matters for hybrid models because reducing Full-Attention KV must not
silently consume memory already promised to recurrent-state pools.

### 3.3 Quantize-on-write

The existing Triton cache-store kernel now supports FP8 destinations.

For one configured dequantization scale `s`:

```text
stored_fp8 = clip(x / s, -448, 448)
restored   = cast(stored_fp8) * s
```

K and V use separate scalar scales.

The cache layout does not change:

```text
[num_blocks, block_size, num_kv_heads, head_dim]
```

Therefore:

- block tables do not change;
- prefix-cache block references do not change;
- preemption ownership does not change;
- slot mapping does not change.

Only the payload dtype and scale metadata change.

### 3.4 Why scale 1.0 is allowed but warned

Both vLLM and SGLang support FP8 KV configurations where missing calibrated
scales fall back to 1.0, while warning that accuracy can degrade.

This project follows the same useful development behavior: a default scale
makes the storage path testable, but the runtime emits a warning when FP8 is
enabled with uncalibrated K/V scales.

For a real benchmark, scale calibration must be treated as part of the
experiment rather than hidden inside the runtime.

References:

- vLLM Quantized KV Cache:
  https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- SGLang Quantized KV Cache:
  https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx

### 3.5 Read path: correctness first

The current `flash-attn` dependency used by nano-vLLM is not assumed to expose
a stable fused FP8-paged-cache API across all supported installations.

The FP8 path therefore keeps the persistent cache quantized and materializes
only the live pages for the current request into the query compute dtype before
calling FlashAttention.

For decode:

```text
paged FP8 KV
   -> gather live request pages
   -> dequantize to compute dtype
   -> FlashAttention GQA
```

For chunked prefill with an existing prefix, the same operation is performed per
packed request. Fresh prefill still attends directly to freshly produced K/V
while writing FP8 copies into the persistent cache.

This is explicitly a reference path. It validates:

- FP8 write semantics;
- paged addressing;
- chunked-prefill continuity;
- prefix-cache reuse;
- preemption/replay compatibility;
- numerical behavior.

It is not presented as the final throughput path.

SGLang's quantized-KV documentation makes the same performance constraint
explicit: unfused dequantization before attention can remove the bandwidth
benefit of compressed KV. That is why this project profiles before adding a
fused decode kernel.

---

## 4. What to validate for FP8

Do not stop at "the model runs."

Recommended validation matrix:

```text
KV dtype:
  BF16
  FP8 E4M3

context:
  1K / 4K / 8K / 16K

execution:
  fresh prefill
  chunked prefill
  decode
  prefix-cache hit
  preemption + replay
```

Record:

- top-1 token agreement;
- max/mean logits error;
- generated-token agreement over a fixed greedy trace;
- persistent KV bytes;
- maximum resident concurrency;
- TTFT and inter-token latency.

Because FP8 is lossy, exact greedy-token equality should be measured, not
assumed.

---

## 5. Profile-gated GQA / PagedAttention optimization

### 5.1 Why no custom kernel was added immediately

A hybrid Qwen3.5 model executes Full Attention only on selected layers. A
beautiful PagedAttention kernel can still have negligible end-to-end impact if
GDN, MoE, or other layers dominate decode.

The runtime now exposes profiler ranges:

```text
nanovllm::decode_model
nanovllm::full_attention_decode
nanovllm::kv_cache_store
```

`profile_decode.py` measures a decode-only window after all requests finish
prefill.

Example:

```bash
python profile_decode.py /path/to/model \
  --context-len 4096 \
  --batch-size 8 \
  --profile-steps 16 \
  --kv-cache-dtype auto
```

FP8 storage path:

```bash
python profile_decode.py /path/to/model \
  --context-len 4096 \
  --batch-size 8 \
  --profile-steps 16 \
  --kv-cache-dtype fp8_e4m3
```

The script prints JSON containing:

- total measured decode GPU time;
- Full-Attention decode GPU time;
- KV-store GPU time;
- Full-Attention fraction;
- whether the configured optimization gate was passed.

The default gate is 15%. This is an engineering threshold, not a universal
constant. It exists to force an explicit decision.

### 5.2 Profile matrix

A useful sweep is:

```text
batch:
  1 / 2 / 4 / 8 / 16

context:
  1K / 4K / 8K / 16K
```

Run both BF16 and FP8 storage.

A custom decode kernel is justified only if the relevant deployment region
shows a material Full-Attention share and the current backend is bandwidth or
dequantization limited.

---

## 6. If the profile gate passes: kernel design

The next kernel should target the actual Qwen GQA geometry rather than imitate a
general production backend.

For a layer with:

```text
num_q_heads = Hq
num_kv_heads = Hkv
group_size = Hq / Hkv
```

the useful work-sharing unit is one request and one KV head.

Conceptually:

```text
grid = request x kv_head

one program:
  load one paged K/V stream
  serve the query heads belonging to that KV head
  dequantize FP8 on load
  update online softmax
  accumulate output
```

The paged address is:

```text
logical_token
  -> logical_block + offset
  -> block_table[request, logical_block]
  -> physical FP8 K/V address
```

The fused path should avoid:

```text
FP8 cache
  -> dense BF16 temporary KV
  -> attention kernel
```

and instead perform:

```text
FP8 global load
  -> scale in register
  -> QK / online softmax / PV
```

### 6.1 What must be benchmarked

A new kernel is accepted only if it beats the current backend over the target
shape matrix. Measure:

- kernel latency;
- end-to-end decode step latency;
- HBM bytes/read efficiency;
- occupancy/register pressure;
- numerical error versus the BF16 reference.

Do not report a kernel microbenchmark speedup as an engine speedup.

### 6.2 Why not copy vLLM or SGLang backend abstractions

vLLM and SGLang support many architectures, quantization formats, devices, and
attention backends. Their backend registries and cache abstractions solve a much
larger problem.

This project borrows three ideas:

- token-budget scheduling should expose a simple resource invariant;
- quantized KV needs explicit scale semantics;
- fused FP8 decode is valuable only when the backend actually consumes the
  compressed cache efficiently.

It keeps a single Qwen3.5-MoE execution path and adds specialization only after
profiling.

Useful references:

- vLLM FP8 KV cache overview:
  https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- SGLang attention backend capability table:
  https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx
- FlashAttention GQA/KV-cache interface:
  https://github.com/Dao-AILab/flash-attention

---

## 7. Tests added

`tests/test_scheduler.py`

- injects a failure after speculative KV growth;
- verifies the new KV tail is released;
- verifies a newly acquired GDN slot is released;
- verifies the request remains retryable from the previous logical boundary.

`tests/test_fp8_kv_cache.py`

- validates `auto` cache dtype behavior;
- validates FP8 E4M3 dtype resolution;
- on CUDA, validates scaled FP8 cache write/dequant round-trip.

The existing scheduler, prefix-cache, preemption, and GDN-state tests continue
to exercise the unchanged ownership rules.

---

## 8. Interview explanation

A compact version:

> The scheduler already had decode-first token budgeting, so I did not rewrite
> it. I fixed the real correctness gap: prefill admission now reserves the GDN
> state slot and KV growth transactionally, rolling back only newly acquired
> resources on failure. Then I added an opt-in FP8 E4M3 paged KV representation
> with fused quantize-on-write and explicit K/V scales. Prefix caching and
> preemption stay unchanged because they own physical blocks rather than care
> about payload dtype. For reads I first kept a correctness backend that
> materializes only live pages and dequantizes before FlashAttention. I then
> added decode profiler ranges and a benchmark gate. I only plan a fused
> GQA-aware PagedAttention kernel if Full Attention is a meaningful fraction of
> end-to-end decode time; otherwise the hybrid model's larger hotspots should be
> optimized first.

Likely follow-up questions:

**Why is atomic reservation necessary if the scheduler is single-threaded?**

Single-threaded scheduling removes races, not exceptions. Allocation failures
or invariant failures can still occur after one resource is acquired. Hybrid
history must never leave the scheduler boundary with only KV or only GDN
ownership.

**Why does FP8 not require changes to prefix-cache keys?**

The cache key describes token history. The prefix entry owns physical KV block
references plus a GDN snapshot at the same boundary. Changing the KV payload
dtype does not change token identity or block ownership.

**Why keep the first FP8 read path unfused?**

It isolates correctness from optimization. The compressed persistent layout,
scale semantics, prefix reuse, and replay can be validated before introducing a
large custom kernel. Profiling then provides evidence for whether fusion is
worth the engineering cost.

**What would make the reference FP8 path slower than BF16?**

It gathers paged FP8 blocks, converts them to the compute dtype, and then runs
attention. That extra materialization can dominate decode. The long-term
performance path must dequantize inside the paged attention kernel.

**What is the strongest reason to stop before writing the custom kernel?**

If Full-Attention decode is a small fraction of total hybrid decode time, even a
large microkernel speedup has a small Amdahl-law effect on end-to-end latency.
