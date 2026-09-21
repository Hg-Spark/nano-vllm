# Qwen3.5 Hybrid Eager Support: Design & Interview Notes

> Branch: `feature/qwen35-hybrid-eager`
>
> Goal: extend nano-vLLM from a pure Transformer decoder runner to a minimal hybrid runner that can execute Qwen3.5 dense text models containing both Full Attention and Gated DeltaNet (GDN) layers.
>
> This stage prioritizes correctness, inspectability and a clear state model. Kernel fusion, CUDA Graph, MoE, vision, hybrid prefix caching and aggressive tensor-parallel sharding are intentionally deferred.

---

## 1. Why this order?

The dependency chain is:

```text
Qwen3.5 checkpoint/config
        |
        v
[1] model semantics are correct
    - checkpoint mapping
    - zero-centered RMSNorm
    - partial RoPE
    - gated full attention
    - GDN math
    - layer_types dispatch
        |
        v
[2] execution semantics are correct
    - one packed batch
    - Full Attention and GDN share one forward pass
    - per-request identity survives TP worker serialization
        |
        v
[3] state lifetime is correct
    - Conv history
    - recurrent matrix state
    - chunked prefill continuity
    - decode continuity
    - preemption reset
    - finish release
```

This order separates three kinds of bugs:

1. **model bug**: wrong parameter mapping or math;
2. **runner bug**: wrong token/sequence dispatch;
3. **state bug**: correct single call, wrong result across calls.

Trying to optimize kernels before separating these failure domains makes numerical debugging much harder.

---

# Stage 1 — Qwen3.5 Text-only Eager adaptation

## 1.1 Scope

Supported in this stage:

- dense Qwen3.5 text backbone (`qwen3_5_text`);
- eager execution;
- Full Attention + GDN hybrid layers;
- chunked prefill and token-by-token decode;
- existing nano-vLLM paged KV cache for Full Attention layers.

Explicitly deferred:

- Qwen3.5 MoE;
- vision tower;
- MTP;
- CUDA Graph for hybrid execution;
- fused GDN kernel;
- recurrent-state-aware prefix caching.

### Why start with dense text-only?

The first engineering objective is to validate the new **hybrid execution semantics**. Adding MoE introduces routing, expert parallelism and a second large source of numerical/performance bugs. Adding vision introduces a second modality and checkpoint namespace.

A narrow first stage gives a reliable reference path that later optimizations can compare against.

---

## 1.2 Config handling

nano-vLLM originally assumes the root Hugging Face config directly describes the decoder.

Qwen3.5 may expose the text decoder through `text_config`, so the runtime now keeps:

- `hf_config`: original root config;
- `model_config`: decoder/text config used by the runner.

For a detected dense Qwen3.5 text model:

- force `enforce_eager = True`;
- temporarily disable prefix cache;
- dispatch to `HybridModelRunner`.

### Why force eager?

GDN mutates per-request recurrent state every forward pass. Existing CUDA Graph capture assumes the decoder's persistent runtime state is represented by fixed graph inputs plus KV cache buffers.

Capturing first and fixing state semantics later would hide state mutation inside graph replay and make lifecycle errors difficult to isolate.

The intended sequence is:

```text
eager correctness
 -> stable state addresses / explicit pool
 -> fused kernels
 -> CUDA Graph capture
```

---

## 1.3 Checkpoint mapping

Qwen3.5 checkpoint names can contain a language-model namespace such as:

```text
model.language_model.layers.0....
```

while the local minimal model uses:

```text
model.layers.0....
```

The loader therefore supports a model-owned `map_weight_name()` callback before applying nano-vLLM's existing packed-weight logic.

The Qwen3.5 mapper:

- strips the language-model wrapper;
- skips vision weights;
- skips MTP weights;
- preserves strict errors for unexpected text parameters.

### Why keep mapping in the model?

Checkpoint naming is a **model-format concern**. Putting Qwen3.5-specific prefixes into the generic loader would make the loader accumulate architecture-specific branches.

The generic loader only defines the extension point. The model owns the mapping rule.

### Why keep strict failures?

Silently ignoring unknown text weights can make the engine start with uninitialized parameters and produce plausible-looking but incorrect output. Unsupported vision/MTP namespaces are explicitly ignored; unexpected decoder weights fail fast.

---

## 1.4 Qwen3.5-specific math

### Zero-centered RMSNorm

Qwen3.5 uses a zero-centered norm parameterization:

```text
y = RMSNorm(x) * (1 + weight)
```

This differs from the existing Qwen3 path where `weight` is initialized around one.

A separate `Qwen3_5RMSNorm` is kept instead of modifying the old layer.

**Reason:** changing the shared RMSNorm risks silently changing Qwen3 behavior.

---

### Partial RoPE

Qwen3.5 rotates only part of each attention head.

The existing rotary layer assumed:

```text
rotary_dim == head_dim
```

It now rotates the first `rotary_dim` channels and passes the tail through unchanged.

**Reason:** partial RoPE is a reusable primitive and belongs in the generic RoPE implementation.

---

### Gated Full Attention

Qwen3.5 Full Attention produces both query and query-side output gate information. The implementation keeps this explicit:

```text
hidden
  -> q_proj -> query + gate
  -> q/k norm
  -> partial RoPE
  -> FlashAttention
  -> sigmoid(gate)
  -> o_proj
```

The existing nano-vLLM Attention backend and paged KV cache remain unchanged.

**Reason:** reuse the proven cache/FlashAttention path and only adapt the model-specific projection semantics.

---

## 1.5 Gated DeltaNet reference path

The first implementation deliberately uses readable PyTorch operations:

1. input projection into Q/K/V channels;
2. stateful causal depthwise Conv1d;
3. Q/K L2 normalization;
4. token-wise recurrent delta update;
5. gated RMSNorm;
6. output projection.

The recurrence is conceptually:

```text
S_t = exp(g_t) * S_(t-1)
prediction_t = k_t^T S_t
delta_t = beta_t * (v_t - prediction_t)
S_t = S_t + k_t * delta_t^T
o_t = q_t^T S_t
```

The recurrent state is kept in fp32 for the reference path.

### Why a Python scan first?

A fused Triton/FLA kernel changes both:

- the execution schedule;
- numerical behavior.

Keeping the recurrence readable provides a reference implementation for later kernel validation:

```text
optimized_kernel_output ~= reference_scan_output
optimized_final_state ~= reference_final_state
```

This is more useful during development and interviews than importing a large opaque backend immediately.

---

# Stage 2 — Hybrid ModelRunner

## 2.1 Problem with the original runner

The original runner assumes every decoder layer consumes the same cache model:

```text
token positions -> KV cache blocks
```

Qwen3.5 contains two layer families:

| Layer | Persistent history |
|---|---|
| Full Attention | K/V tensors indexed by token position |
| GDN | fixed-size recurrent matrix + short Conv1d history |

The model forward can still be one layer loop, but the runner must carry enough request identity for GDN to find its state.

---

## 2.2 Minimal runner abstraction

The base runner now exposes small hooks:

```python
build_model(...)
initialize_runtime_state(...)
prepare_sequence_state(...)
release_sequences(...)
```

`HybridModelRunner` overrides only these hybrid-specific points.

### Why avoid a generic backend framework?

nano-vLLM is valuable because the execution path is small enough to understand.

A vLLM-scale abstraction for cache groups, attention backends and state pools would solve problems this project does not yet have. Four explicit hooks separate the concerns needed today without introducing a hierarchy whose purpose is only future speculation.

---

## 2.3 Packed-batch sequence metadata

Full Attention already understands a packed batch through cumulative sequence lengths.

GDN additionally needs to know which token range belongs to which request. The runtime context now carries:

```text
seq_ids  = (request A, request B, ...)
seq_lens = (tokens for A, tokens for B, ...)
```

A GDN layer slices the packed hidden-state tensor by these boundaries and updates each request's own state.

### Why not store state directly on Sequence?

`Sequence` is scheduler metadata and is serialized to tensor-parallel workers.

Putting CUDA tensors inside it would:

- make IPC serialization expensive;
- mix CPU scheduling state with GPU execution state;
- duplicate or transfer large recurrent tensors.

The sequence carries a stable integer identity. GPU state remains runner-owned.

---

## 2.4 Full Attention KV allocation

Only Full Attention layers need KV cache.

For a hybrid model, KV block memory is therefore sized with:

```text
num_full_attention_layers
```

instead of:

```text
num_hidden_layers
```

This keeps the existing paged KV representation and avoids allocating meaningless KV memory for GDN layers.

---

# Stage 3 — GDN State Manager

## 3.1 State representation

Each active request owns, for each GDN layer:

```text
conv_state      [conv_channels, kernel_size - 1]
recurrent_state [value_heads, key_head_dim, value_head_dim]
```

Lookup key:

```text
(seq_id, layer_idx)
```

### Why separate it from BlockManager?

The two caches have different allocation semantics.

KV cache:

- grows with token count;
- paged into blocks;
- can share immutable prefix blocks.

GDN state:

- fixed size per active request;
- represents the **entire processed prefix**;
- mutates after every token;
- cannot be reconstructed from a KV block id alone.

Forcing both into one block-table abstraction would hide these differences and complicate correctness.

---

## 3.2 Lifecycle

The state lifecycle is:

```text
request admitted
      |
      v
first GDN use -> lazy allocate zero state
      |
      v
chunked prefill -> carry final chunk state to next chunk
      |
      v
decode -> update same state one token at a time
      |
      +---- scheduler preemption/recompute
      |            |
      |            v
      |       reset GDN state
      |       rebuild from prompt
      |
      v
request finished -> release all layer states
```

### First prefill

If `num_cached_tokens == 0`, stale state for that `seq_id` is cleared before execution.

### Chunked prefill

State is **not** cleared between chunks. The next chunk continues from the previous chunk's final Conv/recurrent state.

### Decode

One new token updates exactly the same state.

### Preemption

nano-vLLM's current preemption strategy discards KV blocks and recomputes the request. GDN state must follow the same semantic choice.

When the request returns to prefill with zero cached tokens, its recurrent state is reset and reconstructed from the prompt.

### Completion

The engine explicitly releases the sequence's recurrent states after scheduler completion.

---

# Why hybrid prefix caching is disabled in this stage

A KV prefix hit is insufficient for GDN.

Suppose two requests share a token prefix:

```text
prefix P
   |
   +-- Full Attention history: KV(P)
   |
   +-- GDN history: ConvState(P), RecurrentState(P)
```

Reusing only `KV(P)` gives a partially restored model history.

That failure can be silent: Full Attention layers are correct while GDN layers start from zero or unrelated state.

Therefore the safe stage-1 policy is:

```text
hybrid model -> no prefix reuse
```

A future implementation can re-enable it only when a prefix-cache entry includes compatible recurrent-state checkpoints.

---

# Comparison with vLLM / SGLang

The external projects are used as design evidence, not as code templates.

## vLLM ideas worth learning from

- treats hybrid recurrent models as having cache semantics different from pure KV attention;
- exposes model-specific recurrent-state shape/dtype metadata;
- optimizes Qwen3.5 projections and kernels aggressively.

### What this implementation does differently

nano-vLLM keeps:

- separate readable GDN projections;
- a plain Python recurrent reference;
- a small `GDNStateManager`;
- explicit runner hooks.

This preserves the educational value of the code and creates a baseline before optimization.

---

## SGLang ideas worth learning from

SGLang's hybrid designs separate:

- Full Attention KV storage;
- recurrent-state storage.

That separation validates the core architectural decision used here.

### What this implementation does differently

No general cache-pool framework is introduced yet. State is a dictionary keyed by request/layer because it is enough to prove lifecycle correctness.

A slot-based contiguous pool becomes worthwhile when:

- CUDA Graph requires stable addresses;
- scheduling must reserve recurrent-state capacity;
- batched fused GDN kernels need dense state tensors.

---

# Validation strategy

## Unit invariants included

### 1. Conv state continuity

```text
conv(full_sequence)
==
concat(conv(chunk_1), conv(chunk_2 using state_1))
```

and final states must match.

### 2. Recurrent state continuity

```text
scan(full_sequence)
==
concat(scan(chunk_1), scan(chunk_2 using state_1))
```

and final recurrent states must match.

### 3. State isolation

Mutating request A must not change request B.

### 4. Reset/release

- reset returns a request to zero state;
- release removes all request-owned layer state.

---

## Required GPU numerical test before calling the model adapter complete

The most important integration test is against Hugging Face on the same dense checkpoint.

Recommended deterministic setup:

```text
temperature = 0
batch = 1
eager mode
same dtype
same prompt ids
disable stochastic sampling
```

Compare in this order:

1. embedding output;
2. output after first GDN layer;
3. output after first Full Attention layer;
4. final hidden state;
5. logits;
6. greedy generated token ids.

Debugging from the earliest diverging layer is much faster than comparing only generated text.

Suggested acceptance targets depend on dtype, but token ids should match for a short deterministic smoke test before performance work begins.

---

# Known limitations / intentional debt

1. **GDN tensor parallelism is replicated.**
   - Correctness is prioritized.
   - Memory and compute scale poorly with TP.
   - Next step: shard by compatible head dimension and define collectives explicitly.

2. **GDN recurrence is a Python token loop.**
   - This is the numerical reference path.
   - It will be slow for long prefill.
   - Next step: Triton/FLA fused chunk scan with reference comparison.

3. **Hybrid prefix caching is disabled.**
   - Correct until recurrent checkpoints are cached with prefix metadata.

4. **Hybrid CUDA Graph is disabled.**
   - A contiguous state pool with stable addresses should come first.

5. **No recurrent-state admission control.**
   - The manager exposes `bytes_per_sequence`, but scheduler capacity is still KV-centric.
   - Next step: reserve both KV blocks and recurrent slots before admitting a request.

6. **Dense text-only scope.**
   - MoE, vision and MTP intentionally stay outside this stage.

---

# Recommended next engineering order

```text
A. HF numerical alignment on real Qwen3.5 dense checkpoint
       |
B. contiguous recurrent-state pool + admission accounting
       |
C. GDN TP sharding
       |
D. fused prefill/decode GDN kernels
       |
E. recurrent-state-aware prefix cache
       |
F. CUDA Graph
       |
G. MoE / vision extensions
```

The key rule is to keep one trusted eager reference path while optimizing.

---

# Interview explanation

## 30-second project summary

> I extended nano-vLLM from a pure Transformer runner to support Qwen3.5's hybrid Full Attention + Gated DeltaNet decoder. I first implemented a dense text-only eager reference path and checkpoint mapping, then added a minimal HybridModelRunner that carries request identity through packed batches, and finally introduced a GDN State Manager for convolution and recurrent states. The main engineering issue was that recurrent state has a different lifecycle from paged KV cache, especially under chunked prefill and preemption, so I modeled it separately and temporarily disabled KV-only prefix reuse until recurrent checkpoints can be cached safely.

---

## Common follow-up questions

### Q1. Why does GDN need its own state manager?

Because KV cache is token-indexed and grows with sequence length, while GDN stores a fixed-size mutable summary of the whole prefix plus short convolution history. Their allocation, sharing and reset semantics differ.

### Q2. Why is prefix caching dangerous here?

A token-prefix hit restores Full Attention KV but does not automatically restore the GDN recurrent/conv state for the same prefix. Reusing only one part of model history gives incorrect continuation.

### Q3. Why not copy vLLM's hybrid cache design?

nano-vLLM has a much smaller scope. The minimal state manager makes lifecycle semantics explicit and keeps the implementation understandable. A generalized cache framework is justified later when stable state slots, CUDA Graph and admission control require it.

### Q4. Why Eager first?

It gives a trustworthy numerical reference and keeps mutable recurrent state visible. Kernel fusion and graph capture can then be validated against that reference independently.

### Q5. What happens during preemption?

Current nano-vLLM preemption discards KV and recomputes the sequence. GDN state follows the same policy: reset recurrent/conv state and rebuild it during prefill.

### Q6. What is the next performance bottleneck?

The Python recurrent scan dominates GDN prefill. The next optimization is a fused/chunked scan kernel, followed by GDN TP sharding and a contiguous recurrent-state pool.

### Q7. How would you prove a fused kernel is correct?

Run full-sequence and chunked reference scans, compare both outputs and final states, then compare layer outputs/logits against the eager reference over multiple lengths and dtypes.

---

# Files to read in study order

```text
nanovllm/config.py
    -> model detection and scope

nanovllm/models/qwen3_5.py
    -> model semantics and layer dispatch

nanovllm/layers/gated_delta.py
    -> reference Conv + recurrence

nanovllm/utils/context.py
    -> packed request identity

nanovllm/engine/model_runner.py
    -> HybridModelRunner integration

nanovllm/engine/gdn_state.py
    -> state representation and lifecycle

nanovllm/engine/scheduler.py
nanovllm/engine/block_manager.py
    -> prefix-cache safety policy

tests/test_gdn_reference.py
tests/test_gdn_state_manager.py
    -> invariants
```

---

## What should be emphasized in an interview?

The strongest part of this work is not “I added another model class”.

The engineering story is:

```text
I identified that hybrid recurrent inference invalidates a pure-KV cache assumption,
separated model correctness / execution correctness / state-lifecycle correctness,
built a small eager reference implementation,
and left explicit upgrade points for kernel, TP, cache and graph optimizations.
```

That explanation demonstrates understanding of inference-engine architecture rather than only model transcription.
