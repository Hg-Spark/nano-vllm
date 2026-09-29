# 验证与性能分析

[文档导航](README.md) · 上一章：[调度与混合状态](02_scheduler_cache.md) · 下一章：[设计取舍](04_vllm_sglang_tradeoffs.md)

先确认模型计算与状态恢复可信，再测量瓶颈。本文所有命令在仓库根目录执行，假定已完成[安装与模型准备](00_quickstart.md)，并设置 `NANOVLLM_MODEL`。

## 1. 验证分为哪些层次

| 层次 | 入口 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 合成单元测试 | `tests/unit/` | 调度、所有权、张量形状和局部数值行为符合预期 | 真实检查点一定正确 |
| 真实权重贪心对齐 | `verify_qwen3_5_moe.py` | 给定输入上的 HF 与本实现输出 token 一致 | 所有输入和配置都一致 |
| 联合前缀恢复 | 同脚本 `--check-prefix-resume` | 本实现的无缓存路径与精确状态快照恢复路径生成一致 | 所有输入和设备执行路径都一致 |
| 单图 + Decode Graph | `verify_qwen3_5_moe_multimodal.py` | 视觉展开、mRoPE、Chunk 回填及可选 B=1 Graph 的贪心 token 与 HF 一致 | 多图/视频或所有 batch size 都正确 |
| 逐层诊断 | `diagnose_qwen3_5_moe.py` | 定位所测输入中最早超过误差阈值的 decoder 层 | 自动确定层内哪一个算子有错 |
| 基准与 profile | `bench.py`、`profile_decode.py` | 给定设备和负载下的性能、热点 | 模型数值正确、所有负载都同样快 |

当前[结果目录](../results/README.md)仍以“待验证”标注实验。命令存在、测试覆盖存在、实测通过是三件不同的事。

## 2. 运行单元测试

```bash
python -m pytest tests/unit
```

覆盖内容包括模型/视觉权重映射、三轴 mRoPE、图像 Chunk span、prompt-independent image identity、image-aware Prefix、GDN/MoE Decode 等价路径、Graph fallback/recovery、变长打包、调度、KV/GDN 生命周期和 FP8 KV。CUDA 可用且支持 BF16 时，还会运行 GDN 多 step/slot permutation 与 MoE Decode CUDA BF16 等价检查。

部分用例不执行 GPU 运算，但测试模块导入可能仍依赖 torch、Transformers、Triton 和 FlashInfer。FP8 GPU 用例在无 CUDA 时跳过；“有 CPU 级测试”不意味着未安装项目依赖的纯 CPU 环境能收集整个测试集。详见[测试说明](../tests/README.md)。

## 3. 比较真实权重的生成结果

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-new-tokens 32 \
  --max-model-len 4096
```

脚本先运行 Transformers 参考模型，再运行本实现。HF 使用 BF16、关闭随机采样并取消 EOS 提前结束；本实现使用 `temperature=0.0`、`ignore_eos=True`。比较的是生成部分的 token ID，长度不同也算失败。

输出成功标识为 `PASS: greedy token sequences match exactly`；失败会报告 completion 中首个不一致的索引和两侧 token。该脚本直接使用所给 prompt，不会自动套聊天模板。

默认只比较 16 个新 token。正式记录应覆盖多个中文/英文提示、不同输入长度和更长的 continuation，并保留检查点版本和完整输出。不能用一次短提示通过替代全部验证。

### 单图、Chunked Prefill 与 Decode Graph

```bash
python scripts/verify_qwen3_5_moe_multimodal.py "$NANOVLLM_MODEL" /path/to/image.jpg \
  --max-num-batched-tokens 256 \
  --decode-graph
```

脚本使用官方 `AutoProcessor` 与 HF image-text-to-text 模型生成参考 token；nano-vLLM 使用同一 chat template 和图片。`max_num_batched_tokens` 足够小时，展开后的视觉区间会跨多个 Prefill Chunk；`--decode-graph` 同时启用 B=1 固定地址 Decode Graph。应分别跑一次不带和带 `--decode-graph` 的命令，区分多模态正确性与 Graph 正确性。

## 4. 不一致时做逐层诊断

```bash
python scripts/diagnose_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-probe-tokens 64 \
  --relative-rms-threshold 0.02
```

诊断脚本捕获两侧 decoder 层的 hidden states，输出最大绝对误差、平均绝对误差、RMS 误差和相对 RMS：

```text
relative_rms = RMS(nano - HF) / max(RMS(HF), 1e-12)
```

默认阈值为 `0.02`，是排查起点，不是所有输入通用的精度标准。这里诊断的是截断到 `--max-probe-tokens` 的 prompt 前向；如果偏差只在多步 decode 或快照恢复后出现，还需要针对对应路径增加检查。

| 最早异常附近 | 优先检查 |
| --- | --- |
| 输入与首层 | 分词结果、权重映射、embedding |
| Full Attention | Q/gate 拆分、Q/K norm、partial RoPE、位置和块表 |
| GDN | 卷积历史、衰减、delta 更新、状态槽与前缀长度 |
| MoE | softmax、Top-K 归一化、专家维度、共享专家门控 |
| 末端输出 | 最后 RMSNorm、采样位置、`lm_head` 与采样设置 |

超过阈值的位置给出排查范围，不直接证明这一层是根因，因为前层误差可能继续传播。

## 5. 检查联合前缀恢复

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --max-new-tokens 32 \
  --check-prefix-resume \
  --prefix-block-size 16
```

只有 HF 与本实现的贪心对齐通过后，脚本才继续做前缀探针：

1. 构造 A/B 两个输入，共享一个完整块的 token 前缀，后缀不同。
2. 关闭前缀缓存运行 B，得到基线 token。
3. 在启用缓存的实例中运行 A，使共享边界的 KV/GDN 状态发布到缓存。
4. 运行 B，复用共享 KV 并恢复保持原始状态精度的 GDN 快照。
5. 比较 B 的两条生成序列，并输出快照边界和大小。

第二阶段比较的是“本实现无缓存”与“本实现恢复缓存”，不能单独当作 HF 对齐证明。若没有产生缓存条目，探针会失败，不会把无命中的重复计算当作恢复成功。

记录到[数值对齐结果](../results/parity.md)。当前 snapshot 不再主动降精度；恢复验证仍用于发现状态边界、复制或其他执行路径问题。

## 6. 测量端到端性能

```bash
python scripts/bench.py "$NANOVLLM_MODEL" \
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
| prefill/decode 执行吞吐 | 实际安排的该类 token 总数 / 该类执行阶段总耗时 |
| 峰值显存 | 测量期 `torch.cuda.max_memory_allocated()`，换算 GiB |

执行阶段计时包含 `_run_batch()` 中的相关工作，例如采样、后处理以及可能的前缀快照，不是单个 kernel 的纯 GPU 时间。重算会增加实际执行 token 数。FP8 节省缓存存储也不保证端到端峰值按比例降低。

JSON 中延迟以秒表示，文本摘要会转换为毫秒。`--json` 先打印摘要，再输出单行 JSON，整份标准输出不是一个纯 JSON 文件；重定向时建议保存为日志。只有少量请求时，P99 是很小样本上的插值统计，不能直接当作线上尾延迟结论。

用 `--disable-prefix-cache` 可以关闭前缀条目缓存，但随机输入不代表真实业务的共享前缀分布。结果填入[基线记录](../results/5090_baseline.md)，并补充脚本没有自动记录的依赖版本、检查点 revision 和有效上下文上限。

## 7. FP8 KV 的范围与验证

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

首次 prefill 没有历史块表时可以直接使用本轮未量化 K/V；后续分块与 decode 读取缓存。精度测试需要覆盖这些续算路径，不能只检查第一次 forward。

此功能改变持久 KV 的存储 dtype，不改变专家权重或 GDN 状态格式。读取侧不再物化连续 KV 临时张量。两个 scale 均为 `1.0` 时仍会提示未校准告警；性能收益与数值质量都需要实测。

**现有脚本接口的边界：** `bench.py` 和 `verify_qwen3_5_moe.py` 没有 KV dtype/scale 命令行选项；不能给它们传不存在的参数。`profile_decode.py` 支持 KV dtype，但没有 K/V scale 选项，因此 FP8 profile 使用默认 scale。校准后精度对比需要通过 Python API 或扩展验证脚本明确传参，不能用默认 BF16 验证替代。

## 8. 启动显存定容

持久 KV Cache 不再只依据文本 Prefill warmup 定容。ModelRunner 在 KV 分配前依次触发：

1. 最大活动请求数对应的 Decode MoE 路径，用于初始化并覆盖共享 FlashInfer fused-MoE workspace 的真实 decode 峰值；
2. 以 `max_model_len` 为展开后视觉 token 上界构造的 Vision encoder 网格路径；
3. 最大 `max_num_batched_tokens` 对应的 Prefill 路径。

此外，KV budget 会为最多 `max_num_seqs` 个活动单图请求的视觉 feature tensor 留出显式空间。抢占、失败恢复和请求结束都会清理 request-local GPU visual feature，因此这部分内存不会脱离请求生命周期无限驻留。

Decode CUDA Graph 的 workspace 在 KV 定容前已经存在；capture 本身仍可能需要额外 graph pool。若 capture 在 batch size `B` 发生 OOM，运行时释放该 bucket 及所有更大 bucket、保留已经成功捕获的小 bucket，并继续使用 Eager Decode。Graph 失败不能影响正确性基线。

## 9. 先 profile，再选择 kernel


```bash
python scripts/profile_decode.py "$NANOVLLM_MODEL" \
  --context-len 4096 \
  --batch-size 8 \
  --max-batched-tokens 2048 \
  --warmup-steps 4 \
  --profile-steps 16 \
  --kv-cache-dtype auto \
  --trace results/decode-trace.json
```

同一命令可将 dtype 改为 `fp8_e4m3` 观察存储路径差异，但必须另行检查数值质量。请注意，此脚本预算参数叫 `--max-batched-tokens`，与基准脚本的 `--max-num-batched-tokens` 不同。

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

脚本用 CUDA event 测量窗口时间，并将命名范围的设备时间与之比较。`dominant_profiled_hotspot` 只是在 Full Attention、GDN、MoE 三个范围里选最大者，不能理解为所有开销的完整分类。这些占比也不要求精确相加为 100%。

默认 `--attention-threshold=0.15` 表示 attention 占比达到 15% 后值得试验专用 GQA/paged kernel。这是候选筛选阈值，不是加速收益保证；范围全为零或 trace 异常时，应先排查测量结果。

## 10. 实验矩阵与优化顺序

在显存和模型位置限制允许时，可探索 batch `1/2/4/8/16`、context `1K/4K/8K/16K`、KV `auto/fp8_e4m3`。这些是待测试组合，不是已支持容量的实测承诺。

```text
真实权重对齐 + 前缀恢复验证
  → 固定设备、模型与负载，建立基线
  → 测量 GDN / MoE / Full Attention
  → 优化占比大的路径
  → 重新验证数值和端到端指标
  → 对固定 batch 的 Eager / CUDA Graph 分别做数值与 TPOT 对照
```

GDN Prefill 已优先使用 FlashInfer chunk kernel，Decode MoE 已优先使用共享 workspace 的 FlashInfer CUTLASS fused path；两者都保留参考 fallback。后续应先用 profile 验证这些快路径在目标 SM120 workload 的收益，再决定是否继续定制 kernel；Attention 占比显著时再评估更深的 FP8/Paged Attention 优化。

性能记录见[热点分析模板](../results/profile.md)，验收条件见[路线图](../ROADMAP.md)。
