# Qwen3.5-MoE Runtime — Design Decisions and Reasoning

This document records *why* the runtime is shaped this way. Implementation
details live in `qwen35_moe_eager_design.md`.

---

## 1. Why narrow the project to one model family?

The earlier branch supported original Qwen3 plus dense Qwen3.5 and kept a model
registry, TP-oriented layers, CUDA Graph plumbing and prefix-cache switches.

That surface made the code more general but did not help answer the main
engineering question:

```text
How should a small inference engine execute a model whose history is
paged KV + recurrent state, while its FFN is dynamically routed MoE?
```

Supporting only Qwen3.5-MoE lets the project make every assumption explicit:

```text
text-only
TP=1
eager
non-quantized
hybrid attention/GDN
sparse MoE
```

The reduction is useful because an unsupported feature now fails at config
validation rather than silently travelling through partially relevant branches.

---

## 2. What was taken from Transformers?

Transformers is used as the numerical/model-semantic reference.

Important invariants copied:

- zero-centered RMSNorm;
- gated Full Attention;
- partial RoPE;
- Q/K normalization;
- GDN recurrence/state layout;
- FP32 router softmax;
- Top-K routing followed by Top-K weight normalization;
- packed routed-expert tensors;
- routed experts plus a gated shared expert.

The project does not copy the surrounding library architecture.

Reason: for correctness work, the valuable part is the equation and checkpoint
layout. A large model library's dispatch/config hierarchy is unrelated to a
small inference runtime.

---

## 3. What was learned from vLLM/SGLang?

vLLM and SGLang provide evidence about what becomes important after the
correctness baseline:

- experts should eventually execute through fused/grouped kernels;
- expert routing benefits from compact token-to-expert metadata;
- distributed MoE may require expert placement/EP and collectives;
- recurrent state needs explicit ownership rather than pretending it is KV;
- optimized backends need stable state addresses.

Those engines also contain abstractions needed for a much wider product
surface: backend registries, distributed expert placement, many quantization
formats, multiple cache groups and numerous kernel variants.

The rule here is:

```text
copy the invariant,
not the framework built around many unrelated requirements
```

---

## 4. Why remove the model registry?

With exactly one supported family, a registry converts:

```text
config -> lookup -> one possible class
```

into unnecessary indirection.

The runner now directly constructs `Qwen3_5MoeForCausalLM`, while `Config`
strictly validates the checkpoint family.

A registry becomes justified again only when a second supported model family
has a real implementation and shares enough runtime behavior to warrant it.

---

## 5. Why remove TP abstractions rather than leave them dormant?

The chosen correctness target is TP=1. Keeping TP-aware embedding/linear
classes and NCCL worker lifecycle created two problems:

1. every shape/load path had to retain distributed assumptions;
2. the code looked more capable than it actually was for GDN and MoE.

Qwen3.5-MoE distributed inference also introduces a separate design choice:
tensor parallelism vs expert parallelism. Pretending the old Qwen3 TP path
already solved that would be misleading.

Therefore TP/EP should return together with an explicit distributed design.

---

## 6. Why keep BlockManager?

Even with one model family, Full Attention still uses paged KV whose size grows
with sequence length.

That resource needs different operations from recurrent state:

```text
KV:
allocate pages
append pages
index by token/block
release pages

GDN state:
allocate one fixed slot
mutate in place every token
release one slot
```

So BlockManager remains a real abstraction.

What was removed is only cross-request prefix sharing/hashing, because it is not
safe without matching recurrent-state snapshots.

---

## 7. Why keep StateSlotManager?

Inlining its few lines into Scheduler would save code but mix two ownership
rules.

StateSlotManager expresses a useful invariant:

```text
one admitted request
-> one logical recurrent-state slot
-> same slot index in every GDN layer
```

It also exposes recurrent-state capacity separately from KV capacity, which is
important for admission and memory accounting.

---

## 8. Why keep a unified ModelRunner?

Full Attention and GDN differ in persistent history, but both still execute in
the same decoder loop.

The extra runtime metadata is small:

```text
attention:
slot_mapping / block_tables / context_lens

GDN:
state_slots / packed CPU offsets
```

A `HybridModelRunner` subclass would duplicate model loading, sampling and
prefill/decode preparation. The current single runner makes the data-flow
difference visible without introducing a runner hierarchy.

---

## 9. Why remove cross-request prefix cache but keep block tables?

These are separate concepts.

Unsafe cross-request reuse:

```text
Request B reuses KV(prefix from A)
but has no matching GDN recurrent state
=> inconsistent history
```

Required chunked-prefill behavior:

```text
same request:
chunk 2 must attend to its own KV from chunk 1
```

Block tables remain for the second case. Hashing/ref-counted prefix sharing was
removed.

---

## 10. Why use packed expert parameters directly?

The official checkpoint already stores routed experts as packed 3D tensors.

Keeping the same layout gives:

- direct name/shape validation;
- no load-time repacking;
- fewer model/checkpoint translation rules;
- a natural future input to grouped/fused MoE kernels.

The eager implementation groups tokens by active expert with PyTorch operations
only to establish a numerical oracle.

---

## 11. Why no fused MoE now?

A fused kernel changes several things at once:

- token sorting/dispatch;
- GEMM grouping;
- accumulation order;
- dtype/precision behavior.

If introduced before checkpoint parity, a mismatch could originate from either
model semantics or kernel mechanics.

The staged approach is:

```text
naive packed expert reference
        ->
real checkpoint parity
        ->
profile
        ->
grouped/fused MoE
```

The same principle applies to GDN.

---

## 12. Why keep recurrent state FP32?

The recurrence compounds updates across sequence length. The correctness path
keeps the recurrent matrix FP32 to reduce numerical ambiguity while matching
the reference behavior.

This costs memory, but the runtime explicitly accounts for the state pool before
allocating KV blocks.

Lower-precision recurrent state is an optimization experiment, not a baseline
assumption.

---

## 13. Why move GDN offsets to CPU metadata?

The earlier eager implementation stored packed offsets/state slots as CUDA
tensors and every GDN layer called `.tolist()`.

That creates hidden device synchronization repeatedly through the decoder.

These values originate in the CPU scheduler/runner and are tiny control-plane
metadata, so the reference path now keeps CPU tuples alongside the GPU tensors
needed by FlashAttention.

This is a small specialization with a concrete benefit; it avoids inventing a
general metadata transport framework.

---

## 14. Why skip sampling during partial prefill?

A token sampled before the final prefill chunk is discarded because the prompt
is not yet complete.

Sampling it anyway:

- spends a vocabulary softmax/sample operation;
- consumes RNG in stochastic decoding;
- makes output depend on chunk boundaries.

Only completed prefill requests and decode requests should sample.

---

## 15. Why support multiple EOS ids now?

Qwen3.5 generation configuration can contain multiple stop token ids. Treating
EOS as a single scalar is a model-semantic bug, not an optional feature.

The runtime therefore normalizes generation config EOS into a set at engine
startup.

---

## 16. What should be optimized first?

Do not start with EP or a large backend abstraction.

Recommended order:

```text
1. HF greedy parity on a real Qwen3.5-MoE checkpoint
2. layer-wise probes on first divergence
3. profile eager GDN and eager MoE
4. fused/chunked GDN
5. grouped/fused MoE
6. CUDA Graph once data addresses/control flow are stable
7. TP/EP only for a concrete multi-GPU deployment target
```

Each optimized component should remain testable against the eager numerical
oracle.

---

## 17. Interview framing

The core project story is:

> I specialized nano-vLLM for Qwen3.5-MoE and treated its three execution
> mechanisms separately: paged KV for Full Attention, fixed-size recurrent
> state for Gated DeltaNet, and dynamic Top-K dispatch for sparse MoE. I first
> built readable eager references and strict checkpoint loading, then removed
> generic Qwen3/TP/CUDA-Graph/prefix-cache branches that were not valid for the
> selected target. I kept BlockManager, StateSlotManager and Scheduler because
> they represent genuinely different resource lifetimes. I used Transformers
> for equation/checkpoint parity and vLLM/SGLang to understand future optimized
> invariants, without importing their large backend/EP abstractions before they
> were needed.

The strongest follow-up point is that runtime specialization is not just code
deletion. It makes unsupported assumptions explicit and creates a smaller,
trustworthy numerical baseline for later kernel optimization.
