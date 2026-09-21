# Qwen3.5 Hybrid Eager Support — Design, Migration Decisions & Interview Notes

> Branch: `feature/qwen35-hybrid-eager`
>
> Goal: extend nano-vLLM from a pure Transformer inference path to a small,
> inspectable runtime that can execute Qwen3.5 dense text models containing
> both Full Attention and Gated DeltaNet (GDN) layers.
>
> Priority order: correctness -> state lifecycle -> measurable optimization.
> Fused GDN kernels, GDN tensor parallelism, hybrid prefix checkpoints, MoE,
> vision and hybrid CUDA Graph are deliberately deferred.

---

## 1. Implementation order

The dependency chain is:

```text
Qwen3.5 checkpoint/config
        |
        v
[1] Model semantics
    - checkpoint mapping
    - zero-centered RMSNorm
    - partial RoPE
    - gated Full Attention
    - GDN recurrence
    - layer_types dispatch
        |
        v
[2] Runtime semantics
    - one ModelRunner
    - packed prefill/decode metadata
    - Full Attention uses KV cache
    - GDN uses request state_slot
        |
        v
[3] State lifecycle
    - stable logical slot
    - per-layer Conv state pool
    - per-layer recurrent state pool
    - chunked prefill continuity
    - decode continuity
    - preemption/release/reuse
```

This order separates three failure domains:

1. **model bugs** — wrong math or wrong checkpoint mapping;
2. **runner bugs** — wrong sequence/token metadata;
3. **state bugs** — one forward is correct but multiple forwards diverge.

Kernel optimization is intentionally later because it changes execution order and
numerics at the same time.

---

# 2. Migration decision: how much abstraction is actually needed?

This branch originally used a simple prototype:

```text
(seq_id, layer_idx)
        |
        v
Python dict
        |
        v
ConvState + RecurrentState
```

That prototype was useful because it made the GDN lifecycle easy to understand.
It was not kept as the final runtime contract.

The selected design is:

```text
Sequence
   |
   | state_slot: int
   v
StateSlotManager
   |
   +--------------------------+
   |                          |
   v                          v
Full Attention            GDN layer N
BlockManager              state_pool[state_slot]
   |                       |- conv_state
Paged KV cache             |- recurrent_state
```

## 2.1 Abstractions kept

### StateSlotManager — keep

It is a small allocator, not a general cache framework.

Why it is justified now:

- GPU recurrent tensors should not live inside `Sequence`;
- scheduler admission needs a finite recurrent-state capacity;
- all GDN layers can use the same request-level integer slot;
- slot reuse can be tested explicitly;
- later fused kernels can index dense state tensors directly.

Inlining this logic into `Scheduler` would save only a few lines while mixing
request scheduling and recurrent-state ownership.

### Per-layer contiguous state pools — keep

Each GDN layer owns:

```text
conv_state[num_slots, ...]
recurrent_state[num_slots, ...]
```

This is already useful in eager mode because memory usage becomes predictable
and can be subtracted from the KV-cache budget before block allocation.

Stable addresses are also useful later, but CUDA Graph is not the reason this
design is kept today.

### One unified ModelRunner — keep

A separate `HybridModelRunner` subclass was considered and rejected.

Qwen3.5 needs only a small amount of extra metadata:

- `state_slots` in runtime context;
- GDN state-pool allocation;
- hybrid-aware cache memory accounting.

A new runner hierarchy would duplicate most of the existing path. The current
implementation keeps one `ModelRunner` and lets model/layer capabilities expose
what persistent memory they need.

### Model registry — keep

The registry is intentionally small. It removes Qwen-specific conditionals from
the runner and supports both root architectures and nested `text_config`.

This is a concrete two-model dispatch table, not a plugin framework.

## 2.2 Mature abstractions deliberately rejected or deferred

The branch does **not** add:

- a generalized multi-cache framework;
- cache groups or cache-backend interfaces;
- recurrent prefix checkpoint objects;
- GDN-specific CUDA Graph workspace/capture;
- generic state gather/scatter engines;
- a vLLM-scale attention/backend hierarchy.

These become worthwhile only when there are multiple stateful model families or
multiple optimized backends that actually need the abstraction.

## 2.3 Scheduler split: kept, but not generalized

The scheduler returns decode and prefill groups separately.

Why keep this:

- decode and chunked prefill have different token budgets;
- decode priority avoids long-prefill head-of-line blocking;
- the engine can run decode first and then dynamic-length prefill;
- state slots stay stable across both phases.

Why stop here:

- there is no generic scheduling-policy interface;
- no cache-group abstraction is introduced;
- no hybrid-specific scheduler subclass exists.

---

# Stage 1 — Qwen3.5 dense text eager path

## 3. Scope

Supported:

- dense Qwen3.5 text backbone;
- Full Attention + GDN layers;
- eager hybrid execution;
- chunked prefill;
- token-by-token decode;
- existing paged KV cache for Full Attention;
- recurrent Conv/matrix state for GDN.

Deferred:

- Qwen3.5 MoE;
- vision tower;
- MTP;
- fused GDN kernels;
- GDN TP sharding;
- recurrent-state-aware prefix caching;
- hybrid CUDA Graph.

Dense text-only is the right first target because MoE and vision introduce new
failure domains unrelated to the hybrid recurrent execution model.

---

# 4. Config handling

The runtime retains:

- `hf_config`: root Hugging Face config;
- `text_config`: decoder config used by the text runner.

Hybrid detection comes from `layer_types` containing `linear_attention`.

For hybrid models:

- `enforce_eager = True`;
- prefix reuse is disabled;
- active sequence count is bounded by recurrent state slots.

The default recurrent slot count is intentionally conservative for the Python
reference path. It prevents allocating a large recurrent pool merely because
`max_num_seqs` has a high Transformer-oriented default.

---

# 5. Checkpoint mapping

The text-only runtime must map wrapper checkpoints such as:

```text
model.language_model.layers.0...
```

onto the local text module tree:

```text
model.layers.0...
```

Vision/MTP namespaces are explicitly skipped. Unexpected text weights fail
strictly.

The mapping is declared by the model through prefix metadata while the generic
loader applies the rule.

Design principle:

```text
checkpoint format knowledge belongs to the model
loader owns only generic mapping mechanics
```

Strict failure is important: silently ignoring an unexpected decoder tensor can
produce plausible but numerically invalid output.

---

# 6. Qwen3.5 model semantics

## 6.1 Zero-centered RMSNorm

Qwen3.5 uses:

```text
y = RMSNorm(x) * (1 + weight)
```

A separate Qwen3.5 norm is used so the existing Qwen3 behavior is not changed.

## 6.2 Partial RoPE

Only the first `rotary_dim` channels of each head are rotated.

```text
head = [rotary prefix | pass-through tail]
```

Partial RoPE is implemented in the shared rotary primitive because it is a
reusable mathematical operation rather than a Qwen3.5-only runtime policy.

## 6.3 Gated Full Attention

The Full Attention path is:

```text
hidden
  -> q projection -> query + gate
  -> q/k RMSNorm
  -> partial RoPE
  -> existing Attention/Paged KV
  -> sigmoid(gate)
  -> output projection
```

The existing KV-cache and attention backend remain in use.

---

# Stage 2 — Unified hybrid runtime

## 7. Why the original pure-KV assumption is insufficient

Full Attention persistent history:

```text
K/V tensors indexed by token position
```

GDN persistent history:

```text
fixed-size recurrent matrix
+ short causal-convolution history
```

These histories have different allocation and mutation rules.

The model can still execute both layer types in one decoder loop. The runtime
only needs to provide each request's stable `state_slot`.

---

# 8. Runtime context

Packed prefill/decode context carries:

```text
state_slots = [slot(request A), slot(request B), ...]
```

The GDN layer uses cumulative sequence lengths to locate each packed token range
and `state_slot` to locate persistent recurrent state.

Why not put CUDA tensors in `Sequence`?

- `Sequence` is CPU scheduler metadata;
- it is serialized to TP workers;
- GPU state is large and mutable;
- carrying tensors through scheduler/IPC would couple execution memory to
  control-plane objects.

Only an integer slot crosses that boundary.

---

# 9. Hybrid memory accounting

The runner discovers two kinds of persistent modules:

### KV modules

Modules exposing `k_cache` and `v_cache`.

Only Full Attention layers contribute to KV block memory.

### Stateful modules

Modules exposing:

```text
state_cache_nbytes(num_slots)
allocate_state_cache(num_slots)
```

This small capability contract is kept because the runner needs to reserve
recurrent memory before deciding how many KV blocks fit.

The memory budget is conceptually:

```text
GPU cache budget
  - all GDN state pools
  = memory available for Full Attention KV blocks
```

No more general cache interface is introduced.

---

# Stage 3 — GDN state lifecycle

## 10. State representation

Each GDN layer owns:

```text
conv_state[
    num_slots,
    conv_channels,
    conv_kernel_size
]

recurrent_state[
    num_slots,
    value_heads,
    key_head_dim,
    value_head_dim
]
```

The recurrent matrix remains fp32 in the correctness reference path.

---

# 11. Why state slots are better than the initial dict prototype

The initial dictionary design:

```text
dict[(seq_id, layer_idx)] -> tensors
```

has two strengths:

- very easy to prototype;
- lifecycle is explicit.

Its problems become visible once runtime integration is considered:

1. allocation happens incrementally rather than at admission;
2. total GPU state capacity is difficult to budget;
3. batched kernels would need Python lookup/gather;
4. state addresses are not naturally dense;
5. future graph capture would need a second addressing scheme.

A logical slot solves these without introducing a large framework:

```text
request -> integer slot
layer -> dense tensor pool
```

This is the point where the extra structure pays for itself.

---

# 12. Lifecycle

## 12.1 Admission

The scheduler allocates:

- KV blocks through `BlockManager`;
- one recurrent `state_slot` through `StateSlotManager`.

## 12.2 Fresh prefill

When a request starts from prefix length zero, every GDN layer clears the
physical slot before processing tokens.

This is essential because physical slots are reused.

## 12.3 Chunked prefill

For later chunks:

```text
prefix_len > 0
```

the same slot is retained and its final Conv/recurrent state is used as the
initial state for the next chunk.

## 12.4 Decode

Each new token updates the same per-layer state pool entry.

## 12.5 Preemption

Current nano-vLLM preemption discards KV and recomputes the sequence.

GDN follows the same semantic policy:

```text
preempt
 -> release logical state_slot
 -> request returns to prefill
 -> newly allocated slot is cleared on fresh prefill
 -> state is rebuilt from tokens
```

## 12.6 Completion

Both KV blocks and the logical state slot are released.

The physical state tensor does not need eager zeroing at release because fresh
prefill guarantees zero-before-use. This avoids an unnecessary scheduler-to-GPU
operation while preserving correctness.

---

# 13. GDN reference implementation

The eager implementation keeps the mathematical path readable:

1. Q/K/V projection;
2. stateful depthwise causal Conv1d;
3. Q/K L2 normalization;
4. token-wise gated delta recurrence;
5. gated RMSNorm;
6. output projection.

Conceptually:

```text
S_t = exp(g_t) * S_(t-1)
prediction_t = k_t^T S_t
delta_t = beta_t * (v_t - prediction_t)
S_t = S_t + k_t * delta_t^T
o_t = q_t^T S_t
```

Why use a Python scan first?

It provides a trusted target for later optimization:

```text
optimized_output ~= eager_reference_output
optimized_final_state ~= eager_reference_final_state
```

Importing a large optimized backend immediately would hide the state transition
that this project is intended to understand and optimize.

---

# 14. Prefix cache policy

KV-only prefix reuse is unsafe for a hybrid model.

A reusable prefix P contains:

```text
Full Attention: KV(P)
GDN: ConvState(P) + RecurrentState(P)
```

Restoring only KV(P) creates inconsistent model history.

Therefore:

```text
hybrid model -> prefix cache disabled
```

Re-enabling it requires a prefix entry that restores both histories from the
same token boundary.

---

# 15. CUDA Graph policy

Hybrid execution is forced eager in this stage.

During migration, state-slot CUDA Graph plumbing was intentionally *not*
retained merely as future scaffolding.

The intended order remains:

```text
correct eager lifecycle
 -> fused state-aware GDN kernel
 -> stable batched state access
 -> graph-capture validation
 -> enable hybrid CUDA Graph
```

This keeps current code aligned with current capabilities.

---

# 16. Validation

## Unit invariants

### GDN chunk continuity

```text
GDN(full)
==
concat(
  GDN(chunk1, zero_state),
  GDN(chunk2, state_after_chunk1),
  ...
)
```

Outputs and final states must match.

### Decode continuity

Token-by-token execution must match the same eager recurrence over the full
sequence.

### Slot allocation/reuse

`StateSlotManager` tests:

- allocate;
- exhaustion;
- release;
- reuse.

### Physical state reuse

A reused physical slot is deliberately filled with stale nonzero values. A
fresh prefill must clear it and produce the same output/final state as execution
from explicit zero state.

### Scheduler lifecycle

Tests cover:

- state-slot persistence across chunked prefill;
- state-slot release on preemption;
- hybrid prefix cache disabled;
- shared batch/sequence budget.

### Partial RoPE

The pass-through tail is tested separately from the rotated prefix.

---

# 17. Real-checkpoint validation

Before calling Qwen3.5 support complete, compare the same checkpoint against the
Hugging Face implementation using deterministic greedy generation.

Recommended order when debugging:

1. embedding output;
2. first GDN layer;
3. first Full Attention layer;
4. final hidden state;
5. logits;
6. generated token ids.

Comparing only final text makes it much harder to identify the first divergent
semantic layer.

The repository includes `verify_qwen3_5.py` for end-to-end token comparison.

---

# 18. What was learned from larger engines

vLLM/SGLang are useful as architectural evidence:

- recurrent state is not ordinary KV cache;
- hybrid engines need explicit state ownership;
- optimized kernels benefit from dense/stable state layout.

This project intentionally does not reproduce their generalized cache/backend
frameworks.

The chosen rule is:

```text
copy the invariant,
not the framework built around a much larger product surface
```

---

# 19. Known limitations

1. **GDN TP=1**
   - explicit correctness restriction;
   - next step: shard compatible head dimensions and define collectives.

2. **Python recurrent scan**
   - correct but slow for long prefill;
   - next step: Triton/chunked fused scan.

3. **Hybrid prefix cache disabled**
   - needs recurrent state checkpoints.

4. **Hybrid CUDA Graph disabled**
   - should be enabled only after a graph-safe GDN kernel exists.

5. **Dense text-only**
   - MoE, vision and MTP remain separate stages.

6. **Real GPU/checkpoint numerical alignment still required**
   - unit invariants validate state semantics;
   - they do not replace Hugging Face end-to-end comparison.

---

# 20. Recommended next engineering order

```text
A. real Qwen3.5 checkpoint numerical alignment
        |
B. profile eager GDN prefill/decode
        |
C. fused GDN prefill/decode kernel
        |
D. GDN tensor parallelism
        |
E. recurrent-state prefix checkpoints
        |
F. hybrid CUDA Graph
        |
G. MoE / vision
```

Each optimized stage should retain the eager path as its numerical oracle until
the new path is validated.

---

# 21. Interview summary

> I extended nano-vLLM to support Qwen3.5's Full Attention + Gated DeltaNet
> hybrid decoder. I first built an eager numerical reference, then separated
> paged KV history from fixed-size recurrent state. My first prototype keyed GDN
> state by request/layer in a Python dictionary because it made lifecycle bugs
> easy to debug. During runtime integration I replaced that with a small
> request-level StateSlotManager and per-layer contiguous state pools, because
> the engine needed predictable GPU memory accounting, explicit admission
> capacity and dense state addressing. I deliberately kept one ModelRunner and
> avoided a generalized cache framework. Prefix caching and CUDA Graph remain
> disabled until recurrent-state checkpoints and graph-safe GDN kernels exist.

## Common follow-up: why not keep the dict?

Because after proving correctness, the dict no longer matches the data plane:
GPU state is dense, fixed-size and capacity constrained. A stable integer slot
lets scheduler metadata stay on CPU while every GDN layer directly indexes its
GPU state pool.

## Common follow-up: why not merge recurrent state into BlockManager?

KV grows with sequence length and is paged/shareable by token blocks. GDN state
is fixed-size per active sequence and mutates every token. Their ownership and
reuse semantics are different enough that sharing one allocator would obscure
the important invariant.

## Common follow-up: why no HybridModelRunner?

The hybrid differences are small enough to express as runtime metadata and
stateful-layer capabilities. A subclass would duplicate most of the runner and
make the project harder to read.

## Common follow-up: why eager first?

It exposes state mutation directly and gives later fused kernels a numerical
reference.

---

# 22. Study order

```text
nanovllm/config.py
  -> hybrid detection and scope

nanovllm/models/registry.py
  -> model resolution without runner conditionals

nanovllm/models/qwen3_5.py
  -> model math and layer dispatch

nanovllm/layers/gated_delta_net.py
  -> readable GDN recurrence + physical state pools

nanovllm/engine/state_manager.py
  -> logical request-slot ownership

nanovllm/utils/context.py
  -> state_slot metadata crossing runner -> layer boundary

nanovllm/engine/model_runner.py
  -> KV/state memory accounting and packed metadata

nanovllm/engine/scheduler.py
  -> admission, chunked prefill, preemption and release

tests/test_gated_delta_net.py
tests/test_state_manager.py
tests/test_scheduler.py
tests/test_rotary_embedding.py
  -> correctness invariants

verify_qwen3_5.py
  -> real-checkpoint integration validation
```

The central engineering point is the cache-model transition:

```text
pure Transformer:
request history ~= paged KV

hybrid Qwen3.5:
request history =
    paged KV
  + mutable Conv state
  + mutable recurrent matrix
```

Once that invariant is explicit, the rest of the runtime design becomes much
easier to justify.
