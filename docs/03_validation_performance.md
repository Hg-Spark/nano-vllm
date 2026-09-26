# 验证与性能分析

[文档导航](README.md) · 上一章：[调度与混合状态](02_scheduler_cache.md) · 下一章：[设计取舍](04_vllm_sglang_tradeoffs.md)

先确认模型计算与状态恢复可信，再测量瓶颈。本文所有命令在仓库根目录执行，假定已完成[安装与模型准备](00_quickstart.md)，并设置 `NANOVLLM_MODEL`。

## 1. 验证分为哪些层次

| 层次 | 入口 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 合成单元测试 | `tests/unit/` | 调度、所有权、张量形状和局部数值行为符合预期 | 真实检查点一定正确 |
| 真实权重贪心对齐 | `verify_qwen3_5_moe.py` | 给定输入上的 HF 与本实现输出 token 一致 | 所有输入和配置都一致 |
| 联合前缀恢复 | 同脚本 `--check-prefix-resume` | 本实现的无缓存路径与 BF16 快照恢复路径生成一致 | BF16 快照完全无损 |
| 逐层诊断 | `diagnose_qwen3_5_moe.py` | 定位所测输入中最早超过误差阈值的 decoder 层 | 自动确定层内哪一个算子有错 |
| 基准与 profile | `bench.py`、`profile_decode.py` | 给定设备和负载下的性能、热点 | 模型数值正确、所有负载都同样快 |

当前[结果目录](../results/README.md)仍以“待验证”标注实验。命令存在、测试覆盖存在、实测通过是三件不同的事。

## 2. 运行单元测试

```bash
python -m pytest tests/unit
```

覆盖内容包括模型结构与权重映射、RoPE、采样、GDN 分块连续性、变长打包、状态槽所有权、decode 优先调度、KV 增量分配、联合抢占、失败恢复、请求长度检查、前缀引用计数和 FP8 KV 读写。

部分用例不执行 GPU 运算，但测试模块导入可能仍依赖 torch、Transformers、Triton 和 FlashAttention。FP8 GPU 用例在无 CUDA 时跳过；“有 CPU 级测试”不意味着未安装项目依赖的纯 CPU 环境能收集整个测试集。详见[测试说明](../tests/README.md)。

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

## 5. 检查 BF16 联合前缀恢复

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --max-new-tokens 32 \
  --check-prefix-resume \
  --prefix-block-size 256
```

只有 HF 与本实现的贪心对齐通过后，脚本才继续做前缀探针：

1. 构造 A/B 两个输入，共享一个完整块的 token 前缀，后缀不同。
2. 关闭前缀缓存运行 B，得到基线 token。
3. 在启用缓存的实例中运行 A，使共享边界的 KV/GDN 状态发布到缓存。
4. 运行 B，复用共享 KV 并恢复 BF16 GDN 快照。
5. 比较 B 的两条生成序列，并输出快照边界和大小。

第二阶段比较的是“本实现无缓存”与“本实现恢复缓存”，不能单独当作 HF 对齐证明。若没有产生缓存条目，探针会失败，不会把无命中的重复计算当作恢复成功。

记录到[数值对齐结果](../results/parity.md)。保留 BF16 舍入可能影响续算这一前提；不要把快照描述为无损序列化。

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

[读取路径](../nanovllm/layers/fp8_kv.py)按请求查块表，把有效页面收集成连续张量，转回 query dtype 并乘 scale，然后调用 FlashAttention：

```text
FP8 分页存储 → gather → 转回计算 dtype × scale → Attention
```

首次 prefill 没有历史块表时可以直接使用本轮未量化 K/V；后续分块与 decode 读取缓存。精度测试需要覆盖这些续算路径，不能只检查第一次 forward。

此功能改变持久 KV 的存储 dtype，不改变专家权重或 GDN 状态格式。当前读法有 gather、转换与临时张量成本，因此不承诺加速。两个 scale 均为 `1.0` 时会提示未校准告警。

**现有脚本接口的边界：** `bench.py` 和 `verify_qwen3_5_moe.py` 没有 KV dtype/scale 命令行选项；不能给它们传不存在的参数。`profile_decode.py` 支持 KV dtype，但没有 K/V scale 选项，因此 FP8 profile 使用默认 scale。校准后精度对比需要通过 Python API 或扩展验证脚本明确传参，不能用默认 BF16 验证替代。

## 8. 先 profile，再选择 kernel

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

## 9. 实验矩阵与优化顺序

在显存和模型位置限制允许时，可探索 batch `1/2/4/8/16`、context `1K/4K/8K/16K`、KV `auto/fp8_e4m3`。这些是待测试组合，不是已支持容量的实测承诺。

```text
真实权重对齐 + 前缀恢复验证
  → 固定设备、模型与负载，建立基线
  → 测量 GDN / MoE / Full Attention
  → 优化占比大的路径
  → 重新验证数值和端到端指标
  → 算子与输入布局稳定后再考虑 CUDA Graph
```

GDN 占主导时研究循环/分块 kernel；MoE 占主导时研究 token 分组和融合专家执行；Attention 占比显著时再评估直接读取 FP8 页面、寄存器内反量化与在线 softmax 的实现。以上均为优化方向，当前代码没有完成这些融合路径。

性能记录见[热点分析模板](../results/profile.md)，验收条件见[路线图](../ROADMAP.md)。
