# 源码阅读与讲解指南

[文档导航](README.md) · 上一章：[设计取舍](04_vllm_sglang_tradeoffs.md)

建议先跟踪一个请求，再看多个请求的调度，最后进入层内数学。这样每读一个模块，都能知道它在解决哪个具体问题。

## 1. 沿一次 generate 调用读代码

| 顺序 | 源码入口 | 重点观察 |
| --- | --- | --- |
| 1 | [example.py](../example.py) | 模型路径、聊天模板、生成参数 |
| 2 | [LLM](../nanovllm/llm.py) 与 [LLMEngine](../nanovllm/engine/llm_engine.py) | `LLM` 继承引擎；构造模型、tokenizer、scheduler |
| 3 | [Config](../nanovllm/config.py) | 支持范围、参数检查与上下文上限 |
| 4 | [Sequence](../nanovllm/engine/sequence.py) | 已知 token 和已提交历史的区别 |
| 5 | [Scheduler.schedule()](../nanovllm/engine/scheduler.py) | 本轮安排多少 token，资源从哪里来 |
| 5a | [ScheduledChunk](../nanovllm/engine/schedule.py) | 每个请求本轮的 `[start,end)` 与快照需求 |
| 6 | [ModelRunner.run()](../nanovllm/engine/model_runner.py) | 待恢复状态、prefill/decode 分支、采样位置 |
| 7 | [batch.py](../nanovllm/engine/batch.py) | token、绝对位置、块表、query 偏移与状态槽的对齐 |
| 8 | [模型 forward](../nanovllm/models/qwen3_5_moe.py) | embedding → decoder 层 → norm → 所需位置的 logits |
| 9 | [Sampler](../nanovllm/layers/sampler.py) | 贪心与正温度采样分支 |
| 10 | [Scheduler.postprocess()](../nanovllm/engine/scheduler.py) | 提交边界、发布缓存、追加输出、判断结束、释放资源 |

`generate()`不断执行 `step()`，直到等待队列和运行队列都为空。请求完成时，引擎收集生成部分的 token，最后按请求顺序解码为文本。

## 2. 用五个 token 讲清一次生成

下面仅表示逻辑状态，假设资源充足、没有命中前缀缓存、每轮只处理该请求，prompt 是 `[a,b,c]`，最多生成 2 个 token，且没有提前命中 EOS：

| 时刻 | token_ids | committed_tokens | 本轮区间 | 动作 |
| --- | --- | ---: | ---: | --- |
| 入队 | `[a,b,c]` | 0 | — | 等待执行 |
| 安排 prefill | `[a,b,c]` | 0 | `[0,3)` | 预留 KV 与 GDN 槽 |
| 前向完成 | `[a,b,c]` | 0 | `[0,3)` | 物理状态已覆盖 a/b/c，逻辑尚未提交 |
| 后处理 | `[a,b,c,d]` | 3 | — | 提交 3 个 token，追加预测 d |
| 安排 decode | `[a,b,c,d]` | 3 | `[3,4)` | 输入 d，计算后预测 e |
| 完成请求 | `[a,b,c,d,e]` | 0 | — | 达到输出上限，释放请求资源并重置边界 |

最后一行的 `committed_tokens=0` 来自完成后的资源回收，不表示此前没有计算。e 已经成为返回结果，但这个请求不再继续生成，因此不必再为 e 做一次 forward。

这个例子回答了两个常见问题：为什么 decode 开始时已提交长度比序列长度小 1，以及为什么生成 N 个 token 不一定对应 N 次 decode——首个生成 token 来自最后一次 prefill。

## 3. 再加一个请求，观察打包而不是串接历史

创建 A/B 两个不同长度请求时，阅读 `build_prefill_batch_layout()`，在纸上写出：

```text
每个请求的 start/end
拼接后的 input_ids 与 positions
q_offsets 与 FlashInfer page CSR
各请求 block_table
各请求 state_slot 与 state_prefix_lens
```

再查看 GDN `forward()`怎样按 `q_offsets`切片，Attention 怎样按长度与块表隔离历史。具体数值示例见[第 1 章](01_qwen35_hybrid_runtime.md)。

接着将预算调小，观察长 prompt 跨轮次保持 GDN 槽号不变、KV 块逐步增长，而部分 prefill 不采样。这些行为可以对照 [test_prefill_batch.py](../tests/unit/test_prefill_batch.py) 和 [test_scheduler.py](../tests/unit/test_scheduler.py) 阅读。

## 4. 按资源所有权阅读失败路径

不要只看成功输出，继续沿下面四条路径检查谁持有资源：

| 场景 | 阅读路径 | 应能解释的结果 |
| --- | --- | --- |
| 增量预留失败 | `HybridResources.reserve()` | 只撤销本次新增资源，原有历史可保留 |
| 前向抛出异常 | `LLMEngine._run_batch()` → `recover_failed_step()` | 可能被修改的请求历史必须丢弃，异常向上抛出 |
| KV 不足触发抢占 | `schedule()` → `preempt()` → `_reset_to_waiting()` | 保留 token 列表，释放两种历史，之后恢复或重算 |
| 前缀缓存命中 | `PrefixRuntime.try_restore()` → `restore_pending_states()` | 共享 KV 与独占 GDN 槽恢复到同一边界 |

相关文件：[scheduler.py](../nanovllm/engine/scheduler.py)、[hybrid_resources.py](../nanovllm/engine/hybrid_resources.py)、[llm_engine.py](../nanovllm/engine/llm_engine.py)、[prefix_runtime.py](../nanovllm/engine/prefix_runtime.py)、[cache_runtime.py](../nanovllm/engine/cache_runtime.py)。

特别区分“释放请求拥有的 KV 引用”和“物理 KV 块变空闲”。前缀缓存或其他请求仍持有引用时，物理块不会释放。

## 5. 最后进入模型数学

| 学习主题 | 源码 | 建议对照的测试 |
| --- | --- | --- |
| Top-K 路由、共享专家、权重命名 | [qwen3_5_moe.py](../nanovllm/models/qwen3_5_moe.py) | [test_qwen3_5_moe.py](../tests/unit/test_qwen3_5_moe.py) |
| GDN 卷积与循环更新 | [gated_delta_net.py](../nanovllm/layers/gated_delta_net.py) | [test_gated_delta_net.py](../tests/unit/test_gated_delta_net.py) |
| 部分维度 RoPE | [rotary_embedding.py](../nanovllm/layers/rotary_embedding.py) | [test_rotary_embedding.py](../tests/unit/test_rotary_embedding.py) |
| KV 写入、FlashInfer 分页读取与 FP8 scale | [attention.py](../nanovllm/layers/attention.py)、[model_runner.py](../nanovllm/engine/model_runner.py) | [test_attention.py](../tests/unit/test_attention.py)、[test_fp8_kv_cache.py](../tests/unit/test_fp8_kv_cache.py) |
| 采样 | [sampler.py](../nanovllm/layers/sampler.py) | [test_sampler.py](../tests/unit/test_sampler.py) |

先用小张量解释形状和更新顺序，再看真实模型参数。局部测试用合成配置降低理解成本；真实检查点是否对齐仍需按[第 3 章](03_validation_performance.md)验证。

## 6. 五分钟中文讲解提纲

以下提纲描述当前实现，使用时应根据自己实际完成的工作调整主语，并补充已取得的实验结果。

**第一分钟：项目要解决什么。** 这是一个针对 Qwen3.5-MoE 文本推理的专用运行时。模型混合 Full Attention 和 GDN，再通过 MoE 执行条件计算。工程难点是管理不同类型的请求历史。

**第二分钟：状态怎样组织。** Attention 的 KV 随序列增长，使用分页块和引用计数；GDN 使用每请求独占的槽，里面有卷积状态和 FP32 循环矩阵。`committed_tokens`让两种历史保持共同逻辑边界。

**第三分钟：请求怎样推进。** 调度器先给 decode 分配 token 预算，再给 prefill 安排 chunk。执行成功后才提交进度；部分 chunk 不采样。一次调度可包含两类工作，当前实际分两次前向运行。

**第四分钟：怎样恢复与复用。** 前向失败或抢占会联合释放请求历史，保留 token 以便重算。前缀缓存保存整块 KV 引用和相同边界的 BF16 GDN 快照，命中后两者一起恢复，还要保留至少一个 token 重新计算 logits。

**第五分钟：证据与下一步。** 单元测试检查状态约束，真实权重逐 token 对齐检查输出，前缀探针检查舍入后的续算。当前 GDN/MoE 是 eager 参考实现，优化顺序应由 profile 决定。若实验仍待验证，就明确报告待验证，不能宣称已经达到某个吞吐或完成全面对齐。

## 7. 自测问题

1. 为什么已经生成的最后一个 token 还没有进入历史状态？——它是上一步的预测结果，要在下一步作为输入执行。
2. 为什么不能只恢复 KV？——GDN 也依赖同一前缀的状态，否则两类层读到的历史不一致。
3. 为什么跨过一个完整块边界不等于保存了这个边界？——GDN 已经推进到更晚位置，没有自动保留中途状态。
4. 为什么 MoE 激活参数少仍可能需要很多显存？——当前所有专家权重都常驻，只在计算时选择部分专家。
5. 为什么 FP8 KV 不保证更快？——存储更省显存，但 FlashInfer 的具体 kernel、scale 处理与 workload 会影响速度，仍需端到端测量。
6. 为什么要同时报告失败路径？——正确输出只是一个场景，资源复用和异常后的历史一致性同样决定运行时是否可信。
