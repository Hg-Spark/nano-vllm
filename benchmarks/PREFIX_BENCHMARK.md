# Joint KV/GDN Prefix Cache benchmark

Run against the `hybrid-runtime` codebase with a supported Qwen3.5 model and enough GPU memory:

```bash
# Full cache A/B suite; use REPEATS=1 only to smoke-test the harness.
python -m benchmarks.run_tests --config configs/tests.toml --stage prefix
# Or: NANOVLLM_MODEL=/path/to/model REPEATS=5 bash benchmarks/run.sh prefix
```

The specialized `prefix` stage is opt-in: `all` still runs the pre-existing tests without the 8K-prefix matrix. Every configuration compares `--cache-entries 0` to a cache-enabled run with the same random seed, prompt tokens, total input length, batch, model and output length. Every sample runs in a new process; `summarize_bench.py` requires paired token hashes and at least five independent repeats for formal comparisons.

## Workloads

| Workload | Shared prefix (tokens) | Suffix (tokens) | Active target batch | Cache entries |
|---|---|---|---|---|
| Hot | 512, 2048, 8192 | 256 | 1 and 4 | 0 versus 32 |
| Shared fraction (fixed 4096-token prompt) | 0, 1024, 2048, 3072 | 4096 minus prefix | 1 | 0 versus 32 |
| Eviction and fallback | 2048 | 256 | 1 (6 sequential targets) | 0 versus 2 |

Each shared-prefix run uses a warm-up request that stops just past the candidate prefix boundary. With zero shared prefix the warm-up has no common token blocks; all target prompts are independently sampled. Cache-enabled zero-overlap runs must restore zero tokens, while still incurring prefix lookup and GDN snapshot creation overhead during the target run.

The two-entry variant requires at least one actual checkpoint eviction during target execution. The existing prefix-resume correctness verifier is separate from timing measurements.

## Metrics and interpretation

- **TTFT**: measured at engine level, from admission to first generated token. Target requests are measured after warm-up; for active batch >1 this includes scheduler queueing. A/B comparisons use median-of-run mean TTFT.
- **Recovered tokens**: per-request `restored_tokens_per_target`, arithmetic mean, and `restored_prompt_token_ratio = restored prompt tokens / total target prompt tokens`. A request hit rate alone is insufficient to establish actual prefill work saved.
- **Snapshot footprint**: `peak_snapshot_bytes_including_setup` is the logical byte size of *CPU GDN snapshot tensors*, not GPU allocated memory or CPU RSS.
- **Cache miss penalty**: the zero-overlap ON/OFF TTFT comparison isolates the effect of enabling the cache without reuse, including checkpoint/hash/snapshot work. It is not necessarily pure lookup cost.
- **Validation**: output token hashes must match across A/B, no-hit targets must restore zero tokens, and cache-enabled hot workloads must actually restore a prefix.

These are synthetic offline engine measurements with pre-created request tokens and no network or real arrival process; do not report them as production serving P99.

## Joint boundary correctness

Full Attention pages and GDN snapshots must be restored at the same boundary, aligned to `lcm(64, kv_cache_block_size)`. Run targeted GPU correctness checks separately:

```bash
python -m scripts.verify_prefix_resume /path/to/model --prefix-block-size 16 --shared-prefix-len 128 --prefill-budget 65 --max-model-len 1024
python -m scripts.verify_prefix_resume /path/to/model --prefix-block-size 128 --shared-prefix-len 256 --prefill-budget 129 --max-model-len 1024
```

Both scripts compare cache-disabled and restored greedy output, verify the exact shared KV block IDs and *same* GDN snapshot at the restoration boundary, and fail if the expected checkpoint is missing. The non-grid prefill budgets are intended to exercise recovery after truncated prefill chunks. Review failures rather than automatically excluding them.

For a manual small preflight run, use `REPEATS=1`; do not publish A/B speedup figures from that run. The `prefix` stage deliberately requires more KV space for the 8K case and can fail on smaller GPU/model combinations.
