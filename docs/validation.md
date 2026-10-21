# 验证与性能分析

本页集中维护命令、覆盖范围与结果判定。实验报告需保存环境、模型 revision、输入、配置、原始指标及验证状态。模型计算 dtype 必须显式指定为 `bfloat16`，没有 `auto` 默认值。

## 顺序测试脚本

本地模型和 CUDA 环境准备好后，先准备 Python 环境，再通过 [TOML 配置](../configs/tests.toml) 调用 [run_tests.sh](../benchmarks/run_tests.sh)。先修改配置中的 `model` 路径。以下以系统 Python 3.12 为例：

```bash
uv pip install --system --python 3.12 --break-system-packages -r pyproject.toml --extra dev
```

```bash
flashinfer download-kernels
```

```bash
bash benchmarks/run_tests.sh configs/tests.toml
```

测试启动器复用已安装依赖的 `python3`（也可通过 `PYTHON` 指定解释器），不重复安装依赖。[pyproject.toml](../pyproject.toml) 指定依赖和 Linux x86_64 的 PyTorch CUDA 13.0 包源；`--break-system-packages` 仅用于上面的系统 Python 安装示例。模型和匹配的 NVIDIA 驱动需预先准备。

普通依赖默认使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)，PyTorch 使用官方 CUDA 源。临时切换普通依赖的镜像可在运行前设置 `export UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。

默认执行 `check`：GPU 信息采集、单元/集成测试、32-token HF 贪心对齐和前缀恢复对齐。性能测试需显式选择 `bench`、`tail` 或 `capacity`；正式 A/B 对照将 `repeats` 设置为至少 5。各阶段失败即停止，配置、原始 JSON、日志与 `status.txt` 保存在 `results/run_<UTC时间>/`。每次 `check` 都重新执行，不使用历史成功标记跳过。

单元测试和集成测试分别运行，实时显示每个用例，首次失败立即停止并打印简短堆栈，输出同时写入 `unit-tests.log` 和 `integration-tests.log`。`[correctness]` 的 `max_jobs`、`nvcc_threads` 默认均为 1，控制所有阶段的 FlashInfer JIT 编译并发；它们不限制测试请求并发数，也不保证内存峰值低于某个上限。

分阶段执行；单独 bench/tail 不会自动重新进行正确性检查：

```bash
bash benchmarks/run_tests.sh configs/tests.toml check
```

```bash
bash benchmarks/run_tests.sh configs/tests.toml bench
```

```bash
bash benchmarks/run_tests.sh configs/tests.toml tail
```

```bash
bash benchmarks/run_tests.sh configs/tests.toml capacity
```

依赖已安装后，也可以直接使用同一个系统 Python：

```bash
"$(uv python find --system 3.12)" -m benchmarks.run_tests --config configs/tests.toml --stage check
```

```bash
bash benchmarks/run_tests.sh --help
```

模型、精度、GPU、重复次数与各阶段参数统一在 TOML 中修改。路径支持 `~` 和环境变量，相对路径以项目根目录为基准。

`[benchmark].mode` 选择现有 `smoke` 或 `full` 矩阵，具体负载以原始 JSON 为准。A/B 记录建议保留 Git SHA 或源码归档哈希；汇总脚本不要求 Git 元数据；遇到 `PARITY FAILURE` 应保留失败，不绕过输出一致性检查。Tail 的基线为 Step-exclusive Decode-first，容量搜索未启用 Joint Prefix Cache，不能据此填写“前缀缓存使最大并发提升 X%”。

## 常用入口

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5
export NANOVLLM_DTYPE=bfloat16
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
python -m scripts.verify_prefix_resume "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
python -m benchmarks.bench_experiments "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario throughput --out results/throughput.json
python -m scripts.profile_decode "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
```

需要分别记录 PASS / FAIL / SKIP；GDN 固定要求 SM90/100/120 且 K/V head_dim=128，MoE 还受 FlashInfer CUTLASS 架构范围约束。SM75 的 Dense/MoE 均不支持。

## 验证分为哪些层次

| 层次 | 入口 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 合成单元测试 | `tests/unit/` | 调度、所有权、张量形状和局部数值行为符合预期 | 真实检查点一定正确 |
| 真实权重贪心对齐 | `verify_qwen3_5.py` | 给定输入上的 HF 与本实现输出 token 一致 | 所有输入和配置都一致 |
| 联合前缀恢复 | `verify_prefix_resume.py`（独立入口） | 本实现关闭联合前缀复用的基线路径与恢复联合前缀的路径生成一致 | 所有输入和设备执行路径都一致 |
| 单图 + Decode Graph | `verify_qwen3_5_multimodal.py` | 视觉展开、mRoPE、Chunk 回填、贪心 token 与 HF 一致，并在启用 Graph 时检查实际 Replay 计数 | 多图/视频或其他 batch size 都正确 |
| 逐层诊断 | `diagnose_qwen3_5.py` | 定位所测输入中最早超过误差阈值的 decoder 层 | 自动确定层内哪一个算子有错 |
| 基准与 profile | `bench_experiments.py`、`profile_decode.py` | 给定设备和负载下的性能、热点 | 模型数值正确、所有负载都同样快 |

仓库不再提交历史测试结果。每次改动应重新验证真实模型的 HF 对齐、GDN 状态恢复、三轴 mRoPE、Decode Graph 和性能；命令存在、测试覆盖存在、实测通过是三件不同的事。

## 单元与集成测试职责

单元测试使用合成输入/替身检查局部契约，不下载真实权重；收集时仍可能需要 PyTorch、Transformers 和 FlashInfer 依赖。

| 类型 | 主要测试 |
| --- | --- |
| 模型/权重 | `test_qwen3_5_dense.py`、`test_qwen3_5_moe.py` |
| GDN/Attention/FP8 | CUDA: `tests/integration/test_flashinfer_gdn.py`、`test_flashinfer_attention.py`、`test_flashinfer_ops.py` |
| 批次/调度/状态 | `test_prefill_batch.py`、`test_scheduler*.py` |
| 前缀/多模态/Graph | `test_prefix_cache.py`、`test_multimodal.py`、`test_model_runner.py` |
| 测量与实验结果 | `test_experiment_measurement.py`、`test_bench_experiments.py` |
| 真实 GPU/HF 核心 | `tests/integration/test_flashinfer_attention.py`、`test_flashinfer_moe.py`、`test_flashinfer_gdn.py`、`test_flashinfer_rope.py`；模型级 HF 对齐由 `scripts.verify_qwen3_5` 检查 |

CI 分别运行 Ruff/静态检查与自动发现的 CPU 单元测试；FlashInfer GPU Kernel 测试归入 `tests/integration/`，需在支持的 CUDA 设备上运行。CI 通过不代表 GPU 集成测试通过。

CUTLASS MoE 集成测试未在受支持 GPU 上运行时必须保留 PENDING/SKIP；FlashInfer GDN 集成测试同理。现有 GDN 集成测试包含 BF16 的独立逐 token 递推对照及原有批次/状态槽检查；这是测试参考计算，不是运行时 PyTorch GDN 后端。

## 比较真实权重的生成结果

```bash
python -m scripts.verify_qwen3_5 "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-new-tokens 32 \
  --max-model-len 4096
```

脚本先运行 Transformers 参考模型，再运行本实现。HF 使用显式传入的 `--dtype`（与 nano-vLLM 相同）、关闭随机采样并取消 EOS 提前结束；本实现使用 `temperature=0.0`、`ignore_eos=True`。除生成部分的 token ID 严格一致外，还通过两侧 `lm_head` Hook 比较每个生成步骤的原始 logits（包括 Prefill 最后位置与后续 Decode）。逐步计算 `RMS(Nano - HF) / max(RMS(HF), 1e-12)`，默认阈值为 `0.05`，可通过 `--logits-relative-rms-threshold` 修改。结果保存每步相对 RMS 与最大绝对误差，不保存完整词表 logits。

文本、单图对齐和逐层诊断在 MoE 模型上显式指定 HF 的专家后端：`--hf-experts-backend`
接受 `eager`、`batched_mm`、`grouped_mm`，默认 `eager`，不接受 `auto`。
nano-vLLM 在 MoE 上固定使用 FlashInfer CUTLASS Fused MoE，不提供可切换的
MoE 计算后端。HF `eager` 用于独立数值对照，和 FlashInfer 的舍入顺序
可能存在差异；需要比较真实检查点的层输出和生成 token。
部分 Transformers 版本的 `generate()` 可能在 Decode 使用 `batched_mm`，
因此必须记录实际 HF 专家后端。稠密 Qwen3.5 不启用专家后端。
MoE 仅支持 FlashInfer CUTLASS 覆盖的 GPU 架构；不支持 SM75。

token 对齐成功输出 `PASS: greedy token sequences match exactly`，logits 对齐成功输出 `PASS: logits relative RMS`。token 不一致报告首个不同索引，logits 超阈值报告首个异常步骤、相对 RMS 和最大绝对误差；阈值是浮点误差验收标准，并不要求逐元素完全相等。需要定位逐层 Hidden States 误差时，单独运行 `python -m scripts.diagnose_qwen3_5`。该脚本直接使用所给 prompt，不会自动套聊天模板。

默认只比较 16 个新 token。正式记录应覆盖多个中文/英文提示、不同输入长度和更长的 continuation，并保留检查点版本和完整输出。不能用一次短提示通过替代全部验证。

### 单图、Chunked Prefill 与 Decode Graph

```bash
python -m scripts.verify_qwen3_5_multimodal "$NANOVLLM_MODEL" /path/to/image.jpg --dtype "$NANOVLLM_DTYPE" \
  --max-num-batched-tokens 256 \
  --decode-graph
```

脚本使用官方 `AutoProcessor` 与 HF image-text-to-text 模型生成参考 token；nano-vLLM 使用同一 chat template 和图片。`max_num_batched_tokens` 足够小时，展开后的视觉区间会跨多个 Prefill Chunk；`--decode-graph` 同时启用 B=1 固定地址 Decode Graph。应分别跑一次不带和带 `--decode-graph` 的命令。脚本报告请求配置与 token 对齐，并在启用 Graph 时要求至少发生一次实际 Replay；不能仅凭捕获成功或配置开关证明 Graph 路径已执行。

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

默认阈值为 `0.02`，是排查起点，不是所有输入通用的精度标准。空捕获、缺层及 NaN/Inf 都不能通过；两侧形状不同也判为异常。这里诊断的是截断到 `--max-probe-tokens` 的 prompt 前向；如果偏差只在多步 decode 或快照恢复后出现，还需要针对对应路径增加检查。

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

1. 构造 A/B 两个输入，共享至少 64 Token 且对齐 KV 页的 token 前缀，后缀不同。
2. 关闭前缀缓存运行 B，得到基线 token。
3. 在启用缓存的实例中运行 A，使共享边界的 KV/GDN 状态发布到缓存。
4. 运行 B，复用共享 KV 并恢复保持原始状态精度的 GDN 快照。
5. 比较 B 的两条生成序列，并输出快照边界和大小。

第二阶段比较的是“不开启联合前缀复用”与“恢复联合前缀快照”，**两者都使用正常 Paged KV 和持久 GDN State Slot**，并不是已移除的无持久缓存 Warmup。此项不能单独当作 HF 对齐证明。若没有产生缓存条目，探针会失败。该探针在 KV Block Size 为 16/32 时仍构造至少 64 Token 的共享前缀，以匹配固定 GDN Checkpoint 网格。验证脚本已独立 Hook `restore_gdn_state()`，并检查 `restore_boundary` 与共享 KV Block，避免仅凭缓存条目推断真实命中。

将误差指标记录在独立的本地实验结果中。当前 snapshot 不再主动降精度；恢复验证仍用于发现状态边界、复制或其他执行路径问题。

Prefix Cache 的 GPU KV 静态容量会按每请求最多两个选中 Checkpoint 的跨层驻留/捕获栈，以及单层 FlashInfer 中间 Checkpoint 的上界预留工作区；真实峰值仍应以 GPU Profile 验证。Prefix Cache 可同时限制条目数（`max_prefix_cache_entries`）和 CPU GDN 快照总大小（`max_prefix_cache_snapshot_bytes`，单位字节）。字节限制默认为 `None`，即不额外限制；设置为 `0` 时跳过快照发布。初始化时会估算一份 GDN 状态快照的大小；即使预算大于零，只要不足以容纳一份快照，也不执行为该缓存条目额外分片或 GPU→CPU 拷贝。发布时按固定 `snapshot_nbytes` 计账，不重新扫描快照张量；多个快照超过总预算时优先裁剪当前路径的中间 Checkpoint，否则按全局 FIFO 淘汰并释放关联 KV Block 引用。Benchmark 的 `snapshot_bytes_before_target` / `snapshot_bytes_after_target` 直接读取缓存统计。

## 测量端到端性能

```bash
python -m benchmarks.bench_experiments "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario throughput --num-requests 8 --input-len 256 --output-len 32 \
  --max-num-seqs 4 --budget 512 --max-model-len 4096 \
  --out results/throughput.json
```

[bench_experiments.py](../benchmarks/bench_experiments.py) 生成固定长度随机 token 请求并驱动真实 Engine；[bench.py](../scripts/bench.py) 仅负责公共计时与指标计算。预热及模型初始化不计入目标请求耗时。本实验是离线同时到达的合成负载，不包含 HTTP 或网络延迟。

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

JSON 中延迟以秒表示，指标保存在 `--out` 指定的文件中；命令会拒绝覆盖已有结果。只有少量请求时，P99 是小样本插值统计，不能当作线上尾延迟结论。

Prefix Cache 默认关闭；使用 `--scenario prefix --cache-entries 32` 可测量共享前缀，随机输入无法代表真实业务的共享前缀分布。在本地保存基线结果，并补充脚本没有自动记录的依赖版本、检查点 revision 和有效上下文上限。

### 长 Prompt 干扰实验

```bash
python -m benchmarks.bench_experiments "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario interference --num-requests 3 --max-num-seqs 4 \
  --input-len 128 --output-len 32 --budget 512 --max-model-len 4096 \
  --late-input-len 2048 --late-after-decode-steps 4 \
  --out results/interference.json
```

长 Prompt 在短请求完成指定数量的 Decode Step 后入队。结果记录长请求 TTFT 和注入前后的短请求 token 间隔。不同到达模式需分开比较，单次小样本 P99 不能作为线上尾延迟。

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

**现有脚本接口的边界：** `benchmarks.bench_experiments` 支持 `--kv-cache-dtype`，但没有 K/V scale 参数；`verify_qwen3_5.py` 没有 KV dtype/scale 命令行选项。`profile_decode.py` 支持 KV dtype，但没有 K/V scale 选项，因此 FP8 profile 使用默认 scale。校准后精度对比需要通过 Python API 或扩展验证脚本明确传参，不能用其他 dtype 的验证替代。

## 启动显存定容

模型权重、MoE Workspace、FlashInfer Attention Wrapper 和 Decode Graph Workspace 初始化后，ModelRunner 用 **静态预算**分配持久 KV Cache 与每层 GDN State Pool；**不再执行启动显存 Profiling Warmup**。因此是否能够容纳真实高负载，不会在初始化时由模型峰值测量自动证明。

```text
KV_budget = floor(total_vram * gpu_memory_utilization)
            - currently_used_vram - GDN_state_reserve - visual_feature_reserve
KV_blocks = floor(KV_budget / bytes_per_block)
```

`gpu_memory_utilization` 默认 `0.8`，未分给静态缓存的显存由 Prefill、MoE、Vision 临时激活及 CUDA Graph Capture 等共享。这是容量折中，不保证所有形状、路由分布或图像输入都不会 OOM；如出现 OOM，应降低显存利用率、Token Budget 或并发量，而不是把固定余量当作准确测量的峰值。

纯文本模式默认 `max_visual_tokens_per_request=0`，跳过视觉塔加载及视觉特征显存预留；图文请求需要显式配置正整数上限（或 `None` 表示依 `max_model_len` 预留）。启用视觉时，KV 预算按该上限为最多 `max_num_seqs` 个活动单图请求的视觉特征预留 GPU 显存；请求准入强制检查这个上限；Prefill 处理完图像区间后即可清理 request-local GPU 特征；执行异常后当前引擎不可继续使用。Decode CUDA Graph Capture 仍在 KV Cache 分配之后，若已配置 Bucket 捕获时 OOM 或 Plan 失败，初始化直接报错，不自动降级为 Eager。CUDA Graph 的捕获/内部预执行不属于已移除的显存 Profiling Warmup。首次真实请求可能承担算子初始化或编译延迟。

### 运行时显存预算验证（无需新增后端）

KV Cache 仍在初始化时预分配，Scheduler 按需分配逻辑 Block。需要观察启动时的静态 KV 容量计算，可启用 Python 日志：

```bash
python -c 'import logging; logging.basicConfig(level=logging.INFO); from benchmarks.bench_experiments import main; raise SystemExit(main())' \
  "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" --scenario throughput \
  --out results/memory_budget.json
```

`nanovllm.engine.cache_runtime` 会记录静态 KV 预算、KV Block 数和 GDN/Vision 预留；开启 Graph 后 `nanovllm.engine.model_runner` 还会按 Bucket 记录捕获前后空闲显存和 PyTorch 分配器变化。不再读取历史激活峰值。显存预算不能保证覆盖所有输入分布。以下命令用现有脚本测试边界负载，**执行前必须确保模型及配置能容纳相应规模**：

**最大单轮纯文本 Prefill（示例 Token Budget = 1024）：**

```bash
python -m benchmarks.bench_experiments "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --scenario throughput --num-requests 1 --input-len 1024 --output-len 2 \
  --max-num-seqs 1 --budget 1024 --max-model-len 1026 \
  --out results/prefill_limit.json
```

检查 `max_prefill_step_tokens == 1024`，确认实际覆盖了目标上限。

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

`bench_experiments.py` 与 `profile_decode.py` 可观察 PyTorch `allocated` 峰值、`reserved` 峰值和运行结束时的驱动空闲显存（GiB）。`max_prefill_step_tokens` 和 `max_decode_step_tokens` 应与计划的边界负载核对，不可只根据配置参数声称覆盖上限。`bench_experiments.py` 与 `profile_decode.py` 将这些值写入结果，单图验证侧重 HF Token 对齐，不单独报告显存峰值。`free_after` 不是运行中的最小空闲量；`allocated` 与 `reserved` 是 PyTorch 分配器指标，不等于全部驱动级显存使用量。Profiler 自身也可能增加测量开销。测试失败/OOM 必须保留原始异常，不能把未完成的负载记为 PASS。

## Hybrid Benchmark 消融实验

在仓库根目录使用统一脚本执行五类离线实验：固定 Batch Decode、动态吞吐、共享前缀和长 Prompt 干扰。它们直接驱动 Engine，不包含 HTTP 或泊松到达流量；输出延迟不能作为在线服务 SLO 结论。

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5
export NANOVLLM_DTYPE=bfloat16
REPEATS=1 bash benchmarks/run.sh smoke
REPEATS=5 bash benchmarks/run.sh full
REPEATS=5 bash benchmarks/run.sh tail
```

`RESULT_DIR` 可自定义保存位置。Benchmark 不会重复运行 HF 校验；正式性能对照请先运行 `check` 阶段。单组实验可运行 `python -m benchmarks.bench_experiments --help`；仅汇总可运行 `python -m benchmarks.summarize_bench results/<目录> --min-repeats 5`。

| 实验 | A/B 控制 | 观察 |
| --- | --- | --- |
| Decode | 固定 B 下关闭或开启相同 B 的 Graph Bucket | 每步 wall-clock、吞吐与 Replay 次数 |
| Prefill 干扰 | 同调度器的 Mixed 与 Decode-first Step 互斥 | 长请求 TTFT、短请求 Token Gap |
| Joint Prefix | 缓存关闭与缓存开启，目标请求共享相同前缀 | TTFT、命中比例、恢复 Token、Snapshot 存储字节 |
| 动态吞吐 | Eager 与 Graph Bucket | Output tok/s、TPOT、峰值显存 |

### Tail ITL 与完整 Decode 并发边界

`tail` 模式在同一组短请求 Decode 期间注入一个长 Prompt，使用 7 个短请求、每个 512 个输出 token、Batch 上限 8，分别运行 Mixed 与 Step-exclusive Decode-first。每组至少执行 5 次。公共 `scripts.bench.run_benchmark` 记录每次同步观察到的相邻输出 token 间隔（`itl_seconds` / `itl_samples`），`long_prompt_interference.short_token_gap_after_samples` 为长请求注入后的短请求间隔样本数。结果汇总只有在 **每次 A/B 运行至少 1000 个后段间隔样本** 时才打印 Synthetic P99 ITL 降幅。

此处 ITL 的时间戳在 `step()` 返回、GPU 同步后获取，可能包含其他请求的 Prefill、主机调度与同步时间；它是离线合成负载下的观察间隔，不能直接称为线上服务 P99 ITL。对于正式在线 P99 仍需按实际请求到达时间戳测量，并报告分位数的请求/Token 样本量。

通过已有 `benchmarks.bench_experiments --scenario decode` 每次独立启动一个进程，可搜索固定输入输出长度下的完整 Decode 并发上界：

```bash
python -m benchmarks.capacity_search "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --input-len 1024 --output-len 128 --max-batch 32 --budget 512 \
  --out-dir results/capacity_fp16
```

脚本按 Batch Size 二分搜索，要求请求完成、实际最大 Decode Batch 达到目标。每次独立进程执行，保留原始 JSON 和日志。搜索假设固定负载下容量随 Batch Size 单调；`max_verified_batch` 只代表所设上界内的验证结果，达到 `--max-batch` 仍需提高搜索上限才能判断极值。OOM 或已知 KV 容量不足会被视为本轮批次不通过并继续二分搜索；其他未知运行错误会保留日志并中止。FP8 KV 对比还需独立完成数值校验。

固定 Decode 的 `cuda_event_step_window_seconds` 是 CUDA Event 包住 Engine Step 的时间窗口，**包含可能发生的 CPU 提交间隙**；不是纯 Kernel 耗时。测量 GPU Kernel 应使用 `scripts.profile_decode` 或 Nsight。完整生命周期 KV 额度可能降低低显存条件下的并发度，应同时对比吞吐与 TTFT。前缀 Snapshot 存储字节不等于 GPU→CPU 传输耗时。

每组独立进程执行且保留失败结果；汇总先比较同负载的完整输出 Hash，再计算 A/B 百分比。基准使用合成 Token、固定输出长度，小样本 P99 Token Gap 不代表真实在线服务尾延迟。正式结论需要至少五轮并记录 Git SHA、模型 revision、GPU、dtype 和工作负载。

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
| `nanovllm::prefill_model` | Prefill 模型前向范围（Snapshot D2H 当前不在命名区间内） |
| `nanovllm::gdn_prefill_flashinfer` | FlashInfer GDN Prefill |
| `nanovllm::gdn_decode_flashinfer` | FlashInfer GDN Decode |
| `nanovllm::full_attention_prefill` | Full Attention Prefill |
| `nanovllm::ffn_layer` | Dense FFN 层 |

脚本用 CUDA event 测量窗口时间，并将命名范围的设备时间与之比较。`dominant_profiled_hotspot` 只是在 Full Attention、GDN、Dense FFN、MoE 四个范围里选最大者，不能理解为所有开销的完整分类。这些占比也不要求精确相加为 100%。命名范围全部为零时，最大热点为 null，hotspot_evidence 为 unavailable，不能据此选择优化方向。

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

GDN 在 SM90/100/120 且 Head K/V Dimension 均为 128 时固定使用 FlashInfer 0.7.0 的 Indexed Prefill/Decode，原地更新 FP32 `[V,K]` State Pool。需要在支持的 GPU 上运行 `tests/integration/test_flashinfer_gdn.py` 验证批次与状态槽一致性；Prefill TTFT、Decode TPOT、Graph Capture 和显存峰值需实测。MoE Prefill/Decode 都使用 FlashInfer CUTLASS Fused MoE，不提供其他后端或回退。后续先用 profile 验证各路径在目标 workload 的收益，再决定是否继续定制 kernel；Attention 占比显著时再评估更深的 FP8/Paged Attention 优化。

性能记录应包含真实的 GPU Profile、吞吐量、延迟和显存指标；验收条件见[路线图](../ROADMAP.md)。
