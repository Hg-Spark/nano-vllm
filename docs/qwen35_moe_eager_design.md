# Qwen3.5-MoE Eager Runtime Design

## 1. Scope

The runtime supports one execution family:

```text
Qwen3.5-MoE
Text-only
BF16 / non-quantized safetensors
single GPU / TP=1
eager execution
```

The primary reference target is Qwen3.5-35B-A3B.

Unsupported paths are rejected explicitly rather than kept as dormant generic
branches: dense Qwen3/Qwen3.5, vision, MTP, quantization, TP/EP,
cross-request prefix sharing and CUDA Graph.

This scope is deliberate. The project is intended to expose Qwen3.5-MoE's
runtime invariants before adding optimized kernels or distributed execution.

---

## 2. Decoder structure

Each decoder layer has two independent choices:

```text
token mixer:
    Gated DeltaNet
    or
    gated Full Attention

feed-forward:
    Sparse MoE on every layer
```

The model loop therefore stays simple:

```text
hidden
  -> zero-centered RMSNorm
  -> {GDN | Full Attention}
  -> residual
  -> zero-centered RMSNorm
  -> Sparse MoE
  -> residual
```

No separate HybridModelRunner or layer-runner hierarchy is required.

---

## 3. Full Attention path

The Qwen3.5 full-attention block uses:

```text
q_proj -> [query | gate]
k_proj
v_proj
q/k zero-centered RMSNorm
partial RoPE
paged attention
sigmoid(gate) * attention_output
o_proj
```

Only full-attention layers own KV-cache tensors.

The existing FlashAttention-based paged attention backend is retained because
the cache semantics are still token-position KV semantics.

---

## 4. Gated DeltaNet path

GDN owns history that cannot be represented by KV blocks:

```text
ConvState
+
RecurrentState
```

Each GDN layer owns dense state pools:

```text
conv_state[
    state_slot,
    conv_channels,
    conv_kernel_size
]

recurrent_state[
    state_slot,
    value_heads,
    key_head_dim,
    value_head_dim
]
```

The recurrent state remains FP32 in the correctness path.

The eager recurrence is intentionally readable. It is the numerical oracle for
a later fused/chunked kernel.

### 4.1 Request-level state slots

`Sequence` carries a physical state-slot id and one logical hybrid boundary:

```text
state_slot: int
committed_tokens: int
```

`state_slot` selects the physical Conv/Recurrent state row.
`committed_tokens` records the token prefix represented by both GDN state and
paged KV.

`StateSlotManager` owns allocation/release and an explicit
`slot_owners[slot] -> seq_id` mapping. GPU tensors stay inside GDN layers.

One active request uses the same slot index in every GDN layer. The scheduler
advances `committed_tokens` only after both KV and GDN mutations succeed. This
gives:

- predictable state capacity;
- explicit request-to-slot ownership;
- one scheduler-visible source of truth for hybrid history;
- safe physical-slot reuse after finish/preemption;
- a kernel-friendly future layout.

### 4.2 Fresh prefill

A newly admitted request starts with zero GDN state. Physical slots may contain
stale values from earlier requests, so a fresh prefix (`prefix_len == 0`) clears
the selected slot before use.

### 4.3 Chunked and variable-length prefill

Later chunks keep the same state slot:

```text
chunk 1 -> state_1
chunk 2(state_1) -> state_2
...
```

The runner builds a CPU `PrefillBatchLayout` for heterogeneous requests. The
same request index is used across attention offsets, block-table rows, state
slots and committed prefix lengths.

Attention receives Q/K cumulative offsets. GDN only needs the packed query
offsets plus `state_prefix_lens`, which are derived from
`seq.committed_tokens`. This avoids maintaining a duplicate CPU key-offset
view just to recover the recurrent prefix, and GDN still avoids repeated
device-to-host synchronization.

### 4.4 Decode

Every decoded token mutates the same slot in-place.

### 4.5 Preemption

Current preemption follows recompute semantics:

```text
release KV blocks
release recurrent state slot
request returns to waiting
re-prefill from token history
```

No recurrent-state checkpoint is retained.

---

## 5. Sparse MoE path

The correctness implementation follows the checkpoint's native packed expert
layout:

```text
experts.gate_up_proj[
    num_experts,
    2 * moe_intermediate_size,
    hidden_size
]

experts.down_proj[
    num_experts,
    hidden_size,
    moe_intermediate_size
]
```

The router performs:

```text
router_logits = x @ gate.T
router_probs = softmax(router_logits, fp32)
topk_weight, topk_expert = topk(router_probs, k)
topk_weight /= sum(topk_weight)
```

Only active experts are executed. Selected token outputs are weighted and
accumulated with `index_add_`.

This is intentionally not a high-performance dispatch path. It provides a
transparent numerical reference before grouped GEMM/fused MoE work.

### 5.1 Shared expert

The shared expert is a normal SwiGLU MLP:

```text
shared = down(silu(gate(x)) * up(x))
shared *= sigmoid(shared_expert_gate(x))
```

Layer output is:

```text
routed_expert_output + shared_expert_output
```

---

## 6. Checkpoint loading

Checkpoint-format knowledge lives on the Qwen3.5-MoE adapter. The loader itself
only applies declarative mapping rules and performs strict validation.

Wrapper checkpoints store text weights under:

```text
model.language_model.*
```

The local model tree uses:

```text
model.*
```

`Qwen3_5MoeForCausalLM` therefore declares:

1. skip `model.visual.*`;
2. skip `mtp.*`;
3. map `model.language_model.*` to `model.*`.

The generic loader then requires exact parameter names and shapes and fails on
unexpected or missing text weights.

The runtime supports one model family, so `ModelRunner` constructs
`Qwen3_5MoeForCausalLM` directly after `Config` validates the checkpoint
family. This removes a one-entry registry while keeping checkpoint-specific
mapping rules on the model adapter.

The routed expert tensors are already packed in the official checkpoint, so the
loader does not repack per-expert tensors.

---

## 7. Persistent-memory accounting

The runner separates two persistent histories.

### Full Attention

Memory grows with KV blocks:

```text
2 * num_full_attention_layers
  * block_size
  * num_kv_heads
  * head_dim
  * dtype_size
```

### GDN

Memory is fixed per active request:

```text
all layer ConvState pools
+
all layer recurrent matrix pools
```

The runner reserves GDN state memory before computing how many paged KV blocks
fit in the remaining cache budget.

MoE weights are model parameters and are already reflected in ordinary model
memory usage; they are not part of the cache budget.

---

## 8. BlockManager and prefix policy

The eager baseline first reduced `BlockManager` to incremental physical KV
allocation because KV-only cross-request reuse is unsafe for Qwen3.5-MoE: a
reusable prefix also needs matching GDN Conv/Recurrent state.

Stage 9 later adds back only the ownership machinery that the hybrid invariant
requires:

- per-block reference counts;
- full-block shared KV prefixes;
- a bounded exact-match joint prefix cache;
- a GDN snapshot from the exact same token boundary.

Chunked prefill still uses `block_tables` independently for same-request
history. See `qwen35_scheduler_preemption_prefix_design.md` for the joint
cross-request path.

---

## 9. Scheduler semantics

One scheduler step is decode-first:

```text
1. schedule active decode requests
2. spend remaining token/request budget on prefill
```

Admission requires both:

```text
enough KV blocks for the scheduled range
+
one free recurrent state slot
```

KV admission is incremental: a long prompt can start as soon as its current
chunk fits, without reserving blocks for the entire prompt.

`StateSlotManager` validates the request owner of every retained slot.
A partially prefetched request stays at the waiting front and keeps its KV
allocation, state slot and committed state-prefix length. When its final chunk
leaves token budget, a fresh request may join the same variable-length prefill
batch.

### 9.1 Sampling boundary

Partial prefill does not produce a usable next token. The runner therefore
samples only when:

- decode executes; or
- a request completes its final prefill chunk.

This removes wasted vocab sampling and prevents prefill chunk boundaries from
consuming sampling RNG.

### 9.2 EOS handling

Generation configuration may contain more than one EOS token id. The engine
normalizes the generation config into a set and stops on any configured EOS
unless `ignore_eos=True`.

---

## 10. Deliberately removed runtime machinery

Because this runtime is TP=1 and eager-only, it no longer carries:

- NCCL process-group setup;
- multiprocessing TP workers;
- shared-memory RPC;
- TP serialization hooks in `Sequence`;
- CUDA Graph capture/replay state;
- model registry dispatch;
- hybrid/non-hybrid feature switches.

These paths previously increased the number of states that had to be reasoned
about without contributing to the selected target.

---

## 11. Correctness tests

### MoE router

For every token:

```text
number of selected experts == num_experts_per_tok
sum(normalized top-k weights) == 1
```

### Packed experts

The batched expert implementation is compared against a direct token-by-token
reference using the same packed expert weights.

### GDN chunk continuity

```text
GDN(full sequence)
==
concat(GDN(chunk 1), GDN(chunk 2 with saved state), ...)
```

### GDN decode continuity

Token-by-token recurrence must match a single sequential recurrence.

### Slot reuse

Fresh prefill must clear stale physical state.

### Scheduler

Tests cover:

- decode-first budgeting;
- state-slot persistence through partial prefill;
- preemption releasing both KV and state;
- multi-EOS completion.

### Real checkpoint

`verify_qwen3_5_moe.py` compares deterministic greedy token ids against the
Transformers reference.

---

## 12. Known limitations

1. Eager MoE uses Python-visible expert grouping and is slow.
2. GDN recurrence uses a Python token scan and is slow.
3. Only BF16/non-quantized checkpoints are in scope.
4. TP/EP are not implemented.
5. Joint prefix caching is deliberately bounded and full-block aligned; it is
   not a production-scale radix/hash cache.
6. GDN snapshots are synchronous host copies and remain uncompressed.
7. Vision and MTP weights are ignored.
8. CUDA Graph is not enabled.
9. Real GPU/checkpoint parity must be passed before calling support complete.

---

## 13. Next engineering order

```text
A. real checkpoint HF parity
B. layer-wise numerical probes if parity fails
C. profile GDN vs MoE cost
D. fused/chunked GDN prefill + decode
E. grouped/fused MoE expert execution
F. remove remaining Python synchronization from optimized data path
G. CUDA Graph after kernels have stable state addressing
H. TP/EP only if a concrete deployment target requires them
```
