# 验证与性能分析

本页集中维护命令、覆盖范围与结果判定。实验报告使用 [统一模板](../results/TEMPLATE.md)。模型计算 dtype 必须显式指定为 `float16` 或 `bfloat16`，没有 `auto` 默认值。

## 常用入口

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5
export NANOVLLM_DTYPE=float16
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
python -m scripts.verify_prefix_resume "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
python -m scripts.bench "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --json
python -m scripts.profile_decode "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
```

需要分别记录 PASS / FAIL / SKIP；GPU 架构与 FlashInfer CUTLASS MoE 能力决定可以验证的路径。SM75 的 Dense FP16 可运行，MoE CUTLASS 后端不支持 SM75。

## 验证分为哪些层次

| 层次 | 入口 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 合成单元测试 | `tests/unit/` | 调度、所有权、张量形状和局部数值行为符合预期 | 真实检查点一定正确 |
| 真实权重贪心对齐 | `verify_qwen3_5.py` | 给定输入上的 HF 与本实现输出 token 一致 | 所有输入和配置都一致 |
| 联合前缀恢复 | `verify_prefix_resume.py`（独立入口） | 本实现关闭联合前缀复用的基线路径与恢复联合前缀的路径生成一致 | 所有输入和设备执行路径都一致 |
| 单图 + Decode Graph | `verify_qwen3_5_multimodal.py` | 视觉展开、mRoPE、Chunk 回填及可选 B=1 Graph 的贪心 token 与 HF 一致 | 多图/视频或所有 batch size 都正确，或实际执行了 Graph replay |
| 逐层诊断 | `diagnose_qwen3_5.py` | 定位所测输入中最早超过误差阈值的 decoder 层 | 自动确定层内哪一个算子有错 |
| 基准与 profile | `bench.py`、`profile_decode.py` | 给定设备和负载下的性能、热点 | 模型数值正确、所有负载都同样快 |

当前[结果目录](../results/README.md)除 Dense 2B FP16 的历史有限验证外，其他仍以“待验证”标注实验。命令存在、测试覆盖存在、实测通过是三件不同的事。

## 单元与集成测试职责

单元测试使用合成输入/替身检查局部契约，不下载真实权重；收集时仍可能需要 PyTorch、Transformers、Triton 和 FlashInfer 依赖。

| 类型 | 主要测试 |
| --- | --- |
| 模型/权重 | `test_qwen3_5_dense.py`、`test_qwen3_5_moe.py`、`test_config.py` |
| GDN/Attention/FP8 | `test_gdn_chunk.py`、`test_gated_delta_net.py`、`test_attention.py`、`test_fp8_kv_cache.py` |
| 批次/调度/状态 | `test_prefill_batch.py`、`test_scheduler*.py`、`test_hybrid_resources.py` |
| 前缀/多模态/Graph | `test_prefix_cache.py`、`test_multimodal.py`、`test_model_runner.py` |
| 测量/CLI | `test_experiment_*.py`、`test_scripts.py`、`test_script_*.py` |
| 真实 GPU/HF 核心 | `tests/integration/test_flashinfer_attention.py`、`test_flashinfer_moe.py`、`test_hf_gdn_prefill.py` |

无需 GPU 的语法/风格检查：`python -m compileall -q nanovllm scripts tests`、`python -m ruff check nanovllm scripts tests`、`python -m ruff format --check nanovllm scripts tests`。CI 另外运行依赖轻量的 CLI/测量相关单元测试（当前 54 项）与 Markdown 相对链接检查；**这不代表**包含 PyTorch、FlashInfer 的所有单元/集成测试和真实 GPU 负载均已通过。

HF GDN Chunk 的外部对照使用 Transformers 的 `torch_chunk_gated_delta_rule` 与独立 FP64 单 token 递推，不能据此宣称完整模型 logits 对齐。CUTLASS MoE 集成测试未在受支持 GPU 上运行时必须保留 PENDING/SKIP。

## 比较真实权重的生成结果

```bash
python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-new-tokens 32 \
  --max-model-len 4096
```

脚本先运行 Transformers 参考模型，再运行本实现。HF 使用显式传入的 `--dtype`（与 nano-vLLM 相同）、关闭随机采样并取消 EOS 提前结束；本实现使用 `temperature=0.0`、`ignore_eos=True`。比较的是生成部分的 token ID，长度不同也算失败。

文本、单图对齐和逐层诊断在 MoE 模型上显式指定 HF 的专家后端：`--hf-experts-backend`
接受 `eager`、`batched_mm`、`grouped_mm`，默认 `eager`，不接受 `auto`。
nano-vLLM 在 MoE 上固定使用 FlashInfer CUTLASS Fused MoE，不提供可切换的
MoE 计算后端。HF `eager` 用于独立数值对照，和 FlashInfer 的舍入顺序
可能存在差异；需要比较真实检查点的层输出和生成 token。
部分 Transformers 版本的 `generate()` 可能在 Decode 使用 `batched_mm`，
因此必须记录实际 HF 专家后端。稠密 Qwen3.5 不启用专家后端。
MoE 仅支持 FlashInfer CUTLASS 覆盖的 GPU 架构；不支持 SM75。

输出成功标识为 `PASS: greedy token sequences match exactly`；失败会报告 completion 中首个不一致的索引和两侧 token。该脚本直接使用所给 prompt，不会自动套聊天模板。

默认只比较 16 个新 token。正式记录应覆盖多个中文/英文提示、不同输入长度和更长的 continuation，并保留检查点版本和完整输出。不能用一次短提示通过替代全部验证。

### 单图、Chunked Prefill 与 Decode Graph

```bash
python -m scripts.verify_qwen3_5_multimodal "$NANOVLLM_MODEL" /path/to/image.jpg --dtype "$NANOVLLM_DTYPE" \
  --max-num-batched-tokens 256 \
  --decode-graph
```

脚本使用官方 `AutoProcessor` 与 HF image-text-to-text 模型生成参考 token；nano-vLLM 使用同一 chat template 和图片。`max_num_batched_tokens` 足够小时，展开后的视觉区间会跨多个 Prefill Chunk；`--decode-graph` 同时启用 B=1 固定地址 Decode Graph。应分别跑一次不带和带 `--decode-graph` 的命令。脚本报告请求配置与 token 对齐；实际 Graph replay 未独立确认，不能仅凭开关证明 Graph 路径已执行。

## 不一致时做逐层诊断

```bash
python -m scripts.diagnose_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-probe-tokens 64 \
  --relative-rms-threshold 0.02
```

诊断脚本捕获两侧 decoder 层的 hidden states，输出最大绝对误差、平均绝对误差、RMS 误差和相对 RMS：

```text
relative_rms = RMS(nano - HF) / max(RMS(HF), 1e-12)
```

默认阈值为 `0.02`，是排查起点，不是所有输入通用的精度标准。空捕获、缺层、重复捕获及 NaN/Inf 都不能通过；两侧形状不同也判为异常。这里诊断的是截断到 `--max-probe-tokens` 的 prompt 前向；如果偏差只在多步 decode 或快照恢复后出现，还需要针对对应路径增加检查。

| 最早异常附近 | 优先检查 |
| --- | --- |
| 输入与首层 | 分词结果、权重映射、embedding |
| Full Attention | Q/gate 拆分、Q/K norm、partial RoPE、位置和块表 |
| GDN | 卷积历史、衰减、delta 更新、状态槽与前缀长度 |
| MoE | softmax、Top-K 归一化、专家维度、共享专家门控 |
| 末端输出 | 最后 RMSNorm、采样位置、`lm_head` 与采样设置 |

超过阈值的位置给出排查范围，不直接证明这一层是根因，因为前层误差可能继续传播。

## 检查联合前缀恢复

先运行 HF 贪心对齐，再独立检查**关闭联合前缀复用**与**启用联合前缀恢复**时的贪心续算：

```bash
python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --max-new-tokens 32 --dtype "$NANOVLLM_DTYPE"
python -m scripts.verify_prefix_resume "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --max-new-tokens 32 --prefix-block-size 16
```

前缀探针按如下流程执行：

1. 构造 A/B 两个输入，共享一个完整块的 token 前缀，后缀不同。
2. 关闭前缀缓存运行 B，得到基线 token。
3. 在启用缓存的实例中运行 A，使共享边界的 KV/GDN 状态发布到缓存。
4. 运行 B，复用共享 KV 并恢复保持原始状态精度的 GDN 快照。
5. 比较 B 的两条生成序列，并输出快照边界和大小。

第二阶段比较的是“不开启联合前缀复用”与“恢复联合前缀快照”，**两者都使用正常 Paged KV 和持久 GDN State Slot**，并不是已移除的无持久缓存 Warmup。此项不能单独当作 HF 对齐证明。若没有产生缓存条目，探针会失败。缓存条目存在仍不等于 B 实际恢复了状态；结果将 actual_restore 标注为未独立确认，避免把潜在重复计算误报为恢复路径已覆盖。

记录到[数值对齐结果](../results/TEMPLATE.md)。当前 snapshot 不再主动降精度；恢复验证仍用于发现状态边界、复制或其他执行路径问题。

## 测量端到端性能

```bash
python -m scripts.bench "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --num-requests 8 \
  --min-input-len 64 \
  --max-input-len 256 \
  --output-len 32 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --max-model-len 4096 \
  --seed 0 \
  --json
```

[bench.py](../scripts/bench.py)生成随机 token 请求，在开始计时前全部入队，并忽略 EOS 以生成固定长度。模型初始化不计入请求耗时。这是离线同时到达的一批合成请求，没有 HTTP、网络或真实流量到达过程。

### 指标口径

| 指标 | 脚本中的定义 |
| --- | --- |
| 请求吞吐 | 完成请求数 / 整批耗时，单位 req/s |
| 输出吞吐 | 总生成 token 数 / 整批耗时，单位 tok/s |
| 总 token 吞吐 | 输入加输出 token 数 / 整批耗时 |
| TTFT | 统一计时起点到该请求首次观察到生成 token，包含排队等待 |
| TPOT | `(完成时间 - 首 token 时间) / (输出 token 数 - 1)`；只统计输出大于 1 的请求 |
| E2E | 统一起点到该请求完成的延迟 |
| prefill/decode 主机调用吞吐 | 实际安排的该类 token 总数 / 对应 Forward 的主机端调用耗时，字段为 `prefill_host_call_tok_s`、`decode_host_call_tok_s` |
| 峰值显存 | 测量期 `torch.cuda.max_memory_allocated()`，换算 GiB |

TTFT、TPOT、E2E 在 `step()` 返回并同步 GPU 后记录，因此包含同一轮中 Decode 后续的 Prefill 阻塞。主机调用吞吐来自 `_run_batch()` 的 `StepStats`，在异步 CUDA 执行时不代表精确 GPU Kernel 耗时；需要分阶段 GPU 性能时，应使用 CUDA Events 或 Profiler。重算会增加实际安排的 token 数。FP8 节省缓存存储也不保证端到端峰值按比例降低。

JSON 中延迟以秒表示，文本摘要会转换为毫秒。`--json` 先打印摘要，再输出单行 JSON，整份标准输出不是一个纯 JSON 文件；重定向时建议保存为日志。只有少量请求时，P99 是很小样本上的插值统计，不能直接当作线上尾延迟结论。

用 `--disable-prefix-cache` 可以关闭前缀条目缓存，但随机输入不代表真实业务的共享前缀分布。结果填入[基线记录](../results/TEMPLATE.md)，并补充脚本没有自动记录的依赖版本、检查点 revision 和有效上下文上限。

### 长 Prompt 干扰实验

可以使用同一个 `bench.py` 在短请求进入 Decode 后注入一个长 Prompt，
观察 Decode-first Chunked Prefill 对交互延迟的影响：

```bash
python -m scripts.bench "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --num-requests 3 --max-num-seqs 4 \
  --min-input-len 64 --max-input-len 128 --output-len 32 \
  --max-num-batched-tokens 512 --max-model-len 4096 \
  --late-long-input-len 2048 --late-after-decode-steps 4 \
  --disable-prefix-cache --json
```

`--late-long-input-len=0` 为原有的同时到达基准；
指定长输入时，在完成相应数量的短请求 Decode Step 后入队一个长请求。
`long_prompt_interference` 中的短请求 Token Gap 分别统计长请求入队前后的
相邻生成 Token 时间差（含同轮其他阶段的执行影响），并报告长请求从入队
到首 Token 的 TTFT。不同到达模式应分别运行对照，不能仅凭两组
Gap 直接推断因果加速；单次极小样本 P99 也不是线上尾延迟估计。
本实验复用正式 Scheduler/Engine，没有额外实现调度策略。

### FlashInfer MoE 真实算子验证

```bash
python -m pytest tests/integration/test_flashinfer_moe.py -q
```

需要 FlashInfer CUTLASS MoE 支持的 CUDA GPU（SM75 跳过）。
使用小规模合成专家权重，对独立 FP32 SwiGLU 公式比较 FP16/BF16 输出，
并测试不同 Token 数、Shared Expert 以及固定形状 CUDA Graph 重放中
更新专家路由的正确性。它补充的是实核覆盖，无法代替完整
Qwen3.5-35B-A3B 检查点的 Greedy 对齐和性能结果。
由于 CUDA Graph、架构和第三方内核限制，未实际运行的用例必须
记录为“待验证”，不能视为通过。

## FP8 KV 的范围与验证

配置入口是 `LLM(..., kv_cache_dtype="fp8_e4m3", kv_cache_k_scale=..., kv_cache_v_scale=...)`。两个 scale 是应用到各层 KV 的标量，当前没有自动校准流程。

[写入路径](../nanovllm/layers/attention.py)的语义是：

```text
K_storage = cast_fp8(clamp(K / K_scale, -448, 448))
V_storage = cast_fp8(clamp(V / V_scale, -448, 448))
```

[读取路径](../nanovllm/layers/attention.py)直接把 FP8 分页 KV 与 `k_scale/v_scale` 交给 FlashInfer wrapper：

```text
FP8 分页存储 → FlashInfer paged attention（按 scale 反量化）
```

真实请求首次 Prefill 就会分配逻辑 KV Block，通过 FlashInfer 读写预分配的物理缓存；不存在无缓存 Attention Warmup。精度测试需要覆盖首次 Prefill、分块续算和 Decode。

此功能改变持久 KV 的存储 dtype，不改变专家权重或 GDN 状态格式。读取侧不再物化连续 KV 临时张量。两个 scale 均为 `1.0` 时仍会提示未校准告警；性能收益与数值质量都需要实测。

**现有脚本接口的边界：** `bench.py` 和 `verify_qwen3_5.py` 没有 KV dtype/scale 命令行选项；不能给它们传不存在的参数。`profile_decode.py` 支持 KV dtype，但没有 K/V scale 选项，因此 FP8 profile 使用默认 scale。校准后精度对比需要通过 Python API 或扩展验证脚本明确传参，不能用其他 dtype 的验证替代。

## 启动显存定容

模型权重、MoE Workspace、FlashInfer Attention Wrapper 和 Decode Graph Workspace 初始化后，ModelRunner 用 **静态预算**分配持久 KV Cache 与每层 GDN State Pool；**不再执行启动显存 Profiling Warmup**。因此是否能够容纳真实高负载，不会在初始化时由模型峰值测量自动证明。

```text
KV_budget = floor(total_vram * gpu_memory_utilization)
            - currently_used_vram - GDN_state_reserve - visual_feature_reserve
KV_blocks = floor(KV_budget / bytes_per_block)
```

`gpu_memory_utilization` 默认 `0.8`，未分给静态缓存的显存由 Prefill、MoE、Vision 临时激活及 CUDA Graph Capture 等共享。这是容量折中，不保证所有形状、路由分布或图像输入都不会 OOM；如出现 OOM，应降低显存利用率、Token Budget 或并发量，而不是把固定余量当作准确测量的峰值。

KV 预算按 `max_visual_tokens_per_request`（未配置时使用 `max_model_len`）为最多 `max_num_seqs` 个活动单图请求的视觉特征预留 GPU 显存；请求准入强制检查这个上限；请求结束或抢占时清理 request-local 特征；执行异常后当前引擎不可继续使用。Decode CUDA Graph Capture 仍在 KV Cache 分配之后，若已配置 Bucket 捕获时 OOM 或 Plan 失败，初始化直接报错，不自动降级为 Eager。CUDA Graph 的捕获/内部预执行不属于已移除的显存 Profiling Warmup。首次真实请求可能承担算子初始化或编译延迟。

### 运行时显存预算验证（无需新增后端）

KV Cache 仍在初始化时预分配，Scheduler 仍按需分配逻辑 Block。需要观察启动时的静态 KV 容量计算，可在运行脚本前启用 Python 日志，例如用以下入口代替普通 `python -m scripts.bench`：

```bash
python -c 'import logging; logging.basicConfig(level=logging.INFO); from scripts.bench import main; raise SystemExit(main())' \
  "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --json
```

`nanovllm.engine.cache_runtime` 会记录静态 KV 预算、KV Block 数和 GDN/Vision 预留；开启 Graph 后 `nanovllm.engine.model_runner` 还会按 Bucket 记录捕获前后空闲显存和 PyTorch 分配器变化。不再读取历史激活峰值。显存预算不能保证覆盖所有输入分布。以下命令用现有脚本测试边界负载，**执行前必须确保模型及配置能容纳相应规模**：

**最大单轮纯文本 Prefill（示例 Token Budget = 1024）：**

```bash
python -m scripts.bench "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --num-requests 1 --min-input-len 1024 --max-input-len 1024 \
  --output-len 2 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --max-model-len 1026 --disable-prefix-cache --json
```

检查 `max_prefill_step_tokens == 1024`，否则本次没有覆盖指定上限。

**最大 Decode Batch（示例并发数 = 8）：**

```bash
python -m scripts.profile_decode "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --batch-size 8 --context-len 1024 --max-num-batched-tokens 1024 \
  --warmup-steps 2 --profile-steps 4
```

`profile_decode` 每步校验实际 Decode Token 数等于配置的 Batch Size；若无法维持会失败，不会把较小的 Batch 冒充目标负载。此脚本的 `--warmup-steps` 仅用于基准测量前的 Decode 预运行，不参与启动 KV 定容。

**单图 Prefill（使用目标模型支持的较大真实图片）：**

```bash
python -m scripts.verify_qwen3_5_multimodal "$NANOVLLM_MODEL" /path/to/image.jpg \
  --dtype "$NANOVLLM_DTYPE" --max-model-len 4096 \
  --max-num-batched-tokens 1024
```

图片经 Processor 展开的视觉 token 数取决于模型与图片尺寸；该命令**不保证**达到视觉 token 上界，应根据实际展开后的长度核实输入覆盖度。

三种实验都可观察 PyTorch `allocated` 峰值、`reserved` 峰值和运行结束时的驱动空闲显存（GiB）。`max_prefill_step_tokens` 和 `max_decode_step_tokens` 应与计划的边界负载核对，不可只根据配置参数声称覆盖上限。`bench.py` 与 `profile_decode.py` 将这些值写入结果，单图验证则打印 Nano 推理阶段的显存摘要。`free_after` 不是运行中的最小空闲量；`allocated` 与 `reserved` 是 PyTorch 分配器指标，不等于全部驱动级显存使用量。Profiler 自身也可能增加测量开销。测试失败/OOM 必须保留原始异常，不能把未完成的负载记为 PASS。

## 先 profile，再选择 kernel

```bash
python -m scripts.profile_decode "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --context-len 4096 \
  --batch-size 8 \
  --max-num-batched-tokens 2048 \
  --warmup-steps 4 \
  --profile-steps 16 \
  --kv-cache-dtype auto \
  --trace results/decode-trace.json
```

同一命令可将 `--kv-cache-dtype` 改为 `fp8_e4m3` 观察存储路径差异，但必须另行检查数值质量。预算参数使用统一名称 `--max-num-batched-tokens`。

`prepare_fixed_decode_batch()` 按请求逐个运行实际 Scheduler 与 `_run_batch()`，暂存已完成 Prefill 的 Running 请求以阻止其提前 Decode，并在成功构造固定批次后将暂存请求放回运行队列；异常直接中止测量并关闭 Engine。逐层诊断通过标准 `LLM.generate()` 执行单请求、单次 Prefill，并用 Forward Hook 捕获层输出；不再绕过 Engine 的提交与 Fail-Fast 流程。

脚本先完成等待中的 prefill，再预热和采集 decode 窗口。提示由同一个 token 重复组成，适合控制形状，不代表真实文本的专家路由分布。输出中的 `batch_size` 是请求配置值；严谨分析还应从 trace 核对测量窗口中的实际活动请求，尤其是长 prefill、请求提前达到输出上限或资源紧张时。

主要观测范围包括：

| 范围 | 含义 |
| --- | --- |
| `nanovllm::decode_model` | decode 模型前向范围 |
| `nanovllm::full_attention_decode` | Full Attention 的 decode 计算路径 |
| `nanovllm::kv_cache_store` | K/V 写入，不包含在上述 attention 读取范围内 |
| `nanovllm::gdn_layer` | GDN 层计算 |
| `nanovllm::moe_layer` | 路由、专家执行与合并 |
| `nanovllm::gdn_snapshot_d2h` | 前缀快照传输，通常不出现在纯 decode 窗口 |

脚本用 CUDA event 测量窗口时间，并将命名范围的设备时间与之比较。`dominant_profiled_hotspot` 只是在 Full Attention、GDN、MoE 三个范围里选最大者，不能理解为所有开销的完整分类。这些占比也不要求精确相加为 100%。命名范围全部为零时，最大热点为 null，hotspot_evidence 为 unavailable，不能据此选择优化方向。

这些测量用于定位候选热点，不能单独决定是否实现专用 kernel。应结合 trace、真实负载和端到端基准验证优化价值。

## 实验矩阵与优化顺序

在显存和模型位置限制允许时，可探索 batch `1/2/4/8/16`、context `1K/4K/8K/16K`、KV `auto/fp8_e4m3`。这些是待测试组合，不是已支持容量的实测承诺。

```text
真实权重对齐 + 前缀恢复验证
  → 固定设备、模型与负载，建立基线
  → 测量 GDN / MoE / Full Attention
  → 优化占比大的路径
  → 重新验证数值和端到端指标
  → 对固定 batch 的 Eager / CUDA Graph 分别做数值与 TPOT 对照
```

GDN 在 SM90+ 且 Head K/V Dimension 均为 128 时走 FlashInfer 0.7.0 的 Indexed Prefill/Decode，原地更新 FP32 `[V,K]` State Pool；其他设备仍用 PyTorch FP32 chunk-64 递推和 Decode。需要在支持的 GPU 上运行 `tests/unit/test_flashinfer_gdn.py` 验证数值一致性；两条路径的 Prefill TTFT、Decode TPOT、Graph Capture 和显存峰值需分别实测。MoE Prefill/Decode 都使用 FlashInfer CUTLASS Fused MoE，不提供其他后端或回退。后续先用 profile 验证各路径在目标 workload 的收益，再决定是否继续定制 kernel；Attention 占比显著时再评估更深的 FP8/Paged Attention 优化。

性能记录见[热点分析模板](../results/TEMPLATE.md)，验收条件见[路线图](../ROADMAP.md)。
