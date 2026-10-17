# 调度、缓存与混合状态

## 一轮调度与一次前向不同

[Scheduler.schedule()](../nanovllm/engine/scheduler.py)返回 `SchedulerOutput`，分别保存 `decode_chunks` 和 `prefill_chunks`。每个不可变的 `ScheduledChunk` 给出请求及本轮 `[start,end)`；两类工作共享 `max_num_batched_tokens` 和请求数量限制。

调度顺序为：

1. 从运行队列中选择 decode 请求，每个请求计入 1 个 token。
2. 选中的请求放到运行队列尾部，下一轮继续轮转。
3. 用剩余预算优先继续已经接纳的 Partial Prefill，再按 FIFO 接纳新请求。
4. 新请求在 Prefill 前按 `ceil((prompt_len + max_tokens - 1) / block_size)` 预留完整生命周期的逻辑 KV 页，同时遵守 GDN Slot 数量上限。

[LLMEngine.step()](../nanovllm/engine/llm_engine.py)随后先执行 decode forward，再执行 prefill forward。**一个调度轮次可以包含两类工作，但当前没有把它们合成一次混合 forward。**

### 预算示例

假设预算为 8 个 token、最多 4 个请求，运行队列有 A/B/C 三个 decode 请求，等待队列有长输入 D：

| 请求 | 本轮工作 | 消费 token | 消费请求名额 |
| --- | --- | ---: | ---: |
| A | decode | 1 | 1 |
| B | decode | 1 | 1 |
| C | decode | 1 | 1 |
| D | prefill chunk | 最多 5 | 1 |

即使 D 有数千个输入 token，本轮也只预留并执行能安排的片段。

如果只有 2 个 token 预算、运行队列为 `[A,B,C]`，第一轮选中 A/B 后队列成为 `[C,A,B]`；下一轮优先轮到 C。这个公平性只针对已经进入 decode 队列的请求，不保证等待中的 prefill 不会被持续的 decode 压力延后。

## 分块 prefill 如何保持连续

本轮请求片段使用半开区间：

```text
start = committed_tokens
q_len = min(尚未计算的 token 数, 剩余预算, KV 可容纳的新增 token 数)
end   = start + q_len
执行 token_ids[start:end]
```

KV 只增长到容纳 `end` 所需的块数；GDN 始终沿用该请求的同一槽号。部分 prefill 的请求仍留在等待队列，并且仍然占有资源。

以 40-token 输入、每轮可用预算 16、块容量 16 为例，假设没有命中缓存且资源充足：

| 轮次 | 输入位置 | 成功后 committed | 请求持有 KV 块数 | GDN 槽 | 是否采样 |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `[0,16)` | 16 | 1 | 同一个槽 | 否 |
| 2 | `[16,32)` | 32 | 2 | 同一个槽 | 否 |
| 3 | `[32,40)` | 40 | 3 | 同一个槽 | 是，生成首个 token |
| 4 | 位置 40 的首个生成 token | 41 | 3 | 同一个槽 | 是，生成第二个 token |

第 3 轮后，`num_tokens=41`，`committed_tokens=40`。部分 chunk 不会生成可交给用户的 token；只有所有已知输入都处理完，`ModelRunner._sample_logits()` 才选中该请求采样。

同一请求本轮只安排一个 chunk；资源充足时调度器继续扫描其他等待请求。已经接纳的请求保持逻辑 KV 额度，KV 页仍按需物理分配。资源充足时多个请求可以并行 Prefill；不足时新请求等待，不会抢占已有请求。

## 预留、执行、提交是三个阶段

```mermaid
flowchart LR
    A[预留：KV 增量与 GDN 槽] --> B[执行：修改物理张量]
    B --> C[需要时捕获 GDN 快照]
    C --> D[需要时发布联合前缀]
    D --> E[提交 committed_tokens 并追加 token]
```

### 资源预留

Scheduler 按全生命周期 KV 页额度及 GDN 槽位准入新请求。已接纳请求按当前 Chunk 按需分配 KV 页，额外的前缀缓存引用可通过 LRU 淘汰回收；额度不足的新请求继续等待。

### 执行成功：推进一个共同边界

`postprocess()`检查批次结果数量与采样结果，在需要时发布联合前缀，随后提交逻辑进度并追加预测 token：

```text
check result count
  → sampling-result validation
  → publish this chunk's KV/GDN prefix if needed
  → committed_tokens = ScheduledChunk.end
  → append sampled token if applicable
```

不提供批次级回滚：后续 Chunk 失败时，前面请求可能已经提交。Engine 会终止后续执行；新生成的 token 尚未被前向处理，不属于刚发布的 KV/GDN 前缀。

## 为什么无需常规抢占

接纳请求时预留其最大生命周期所需 KV 页额度；活动请求预留量之和不能超过物理 KV 总页数。GDN Slot 同样在接纳时占用，请求完成时与 KV 页一并释放。前缀缓存额外持有的 KV 引用在空间紧张时由 LRU 淘汰；共享页的引用计数继续保证安全。

这一策略避免新请求挤占正在处理的 Prefill / Decode 请求，代价是最大输出长度设置较大时并发度更保守。

## 执行失败与预留失败的区别

调度器在新请求准入时约束完整 KV 额度和 GDN Slot，执行时按需分配物理页。若缓存淘汰后仍无法满足已接纳请求的页增长需求，则抛出异常。

模型前向、CUDA Graph 或提交后处理抛出异常时，物理 KV/GDN 状态可能已部分改变。此时 `LLMEngine._run_batch()` 将引擎标记为不可用，原异常直接上抛。之后的 `step()`、`add_request()` 和 `generate()` 均拒绝继续执行，必须重新构造引擎。

一轮调度依旧可能先执行并提交 Decode，再执行 Prefill。若第二阶段失败，已执行的 Decode 输出不会重新交付；整轮操作失败且当前引擎终止使用。该设计删除跨 Step 补发输出和执行失败后的伪事务回滚。

已发布的共享前缀缓存和请求资源仍由原有生命周期负责释放。由于失败时可能存在 CUDA Context 错误，不能将资源释放成功或同引擎重试作为恢复保证。

## 联合前缀缓存保存什么

[PrefixEntry](../nanovllm/engine/prefix_cache.py)作为联合前缀 metadata，包含：

| 内容 | 为什么需要 |
| --- | --- |
| 精确 `token_ids` 元组 | 确认输入前缀一致，不能仅比较文本看起来是否相同 |
| `block_ids` | 共享 Full Attention 的物理 KV 块 |
| `num_tokens` | 给出共同的历史边界 |
| `GDNStateSnapshot` | 保存同一边界处所有 GDN 层的卷积和循环状态 |

[BlockManager](../nanovllm/engine/block_manager.py)负责 KV block 的所有权与引用计数；[PrefixCache](../nanovllm/engine/prefix_cache.py)负责 prefix entry、最长匹配和 LRU 淘汰，Scheduler 直接通过 BlockManager.prefix_cache 访问前缀缓存；GDN snapshot 只是与完整 KV 前缀绑定的附加 metadata。[Scheduler](../nanovllm/engine/scheduler.py)负责 KV/GDN 资源预留、联合恢复及释放。联合前缀缓存默认关闭（`max_prefix_cache_entries=0`），用于显式验证/实验时再开启，避免默认承担 GDN 快照的 D2H 成本。

### 什么时候能发布

自动快照来自 prefill 成功执行后的边界，并且必须满足：

- 前缀缓存已启用，请求有状态槽且本轮安排了 token；
- 边界大于零且不超过原始 prompt 长度；
- **本轮结束位置恰好是完整 KV 块边界**；
- 该精确 token 前缀尚未存在。

例如块容量为 16、20-token Prompt 且启用 Prefix Cache：Scheduler 会先选取 `[0,16)`，在该边界捕获并发布 KV + GDN 快照，再执行 `[16,20)`。不启用缓存时保持 `[0,20)`；已有缓存条目、视觉 Token 边界及 Token Budget 仍受原有约束。只在可以发布新完整块且 `prefix_cache_eager_snapshot=True`（默认）时额外切分，避免每块都执行一次 Forward；设为 `False` 可禁止为快照额外切分，但自然到达完整块边界时仍会发布，长前缀的缓存命中机会可能减少。

整块边界还能避免共享一个将被继续写入的半满块；新请求续写时分配后续块，不会覆盖共享前缀。

### 怎样查找和恢复

缓存用 `OrderedDict` 保存有上限的 LRU 条目。查找会扫描已有条目，比较精确 token 前缀，选择最长命中，并更新其最近使用顺序。

```text
新接纳的请求
  → 查找最长可用前缀
  → 分配独占 GDN 槽
  → 增加共享 KV 块的引用计数
  → committed_tokens = 命中前缀长度
  → pending_state_snapshot = 命中的 GDN 快照
  → ModelRunner 在下一次 prefill 前恢复设备状态
  → 执行未缓存的后缀
```

**至少留下一个已知 token 重算。** 查找上限为 `num_tokens - 1`，因为缓存没有保存下一 token 的 logits。即使整个输入正好与某个缓存条目一致，也不能跳过全部前向后凭空采样。

## 引用计数怎样保证共享 KV 不被提前释放

设块 X 最初由请求 A 持有：

| 操作 | X 的引用数 |
| --- | ---: |
| A 分配 X | 1 |
| 缓存发布包含 X 的前缀 | 2 |
| A 完成并释放自己的引用 | 1 |
| B 命中并附加 X | 2 |
| 缓存淘汰该条目 | 1 |
| B 完成 | 0，块可重新分配 |

缓存淘汰不一定立即产生空闲 KV 块，因为请求或其他缓存条目可能仍在引用它。活动 GDN 槽则不采用这样的共享引用机制：每个槽只属于一个请求，缓存保存的是它的独立快照。

## GDN 快照的精度与空间代价

[GDNStatePool](../nanovllm/layers/gdn_state.py)将活动状态复制到 CPU，并保持继续计算所需的原始精度：卷积状态保持活动缓冲区 dtype，循环矩阵保持 FP32。恢复时要求快照 dtype 与活动槽一致。

```text
运行中：卷积 = 模型 dtype，循环矩阵 = FP32
快照中：卷积 = 原 dtype CPU，循环矩阵 = FP32 CPU
恢复后：卷积 = 原 dtype，循环矩阵 = FP32
```

这样不会因 snapshot 本身引入 BF16 量化误差。代价是循环状态主机占用约为 BF16 方案的两倍；条目上限约束的是缓存项数，不是精确字节数，缓存还会持有 GPU KV 引用并产生设备到主机的快照传输。单份快照大小由启动时的 GDN 状态池计算；如果配置的 CPU 快照预算连一份也容纳不了，调度器不会为发布前缀主动分片或捕获状态，发布时仍检查实际大小和 LRU 总预算。

## 阅读代码时应检查的约束

- 同一执行批次不能有两个请求使用同一 GDN 槽。
- 有已提交历史时，必须能找到对应 KV 与 GDN 资源或待恢复快照。
- decode 前满足 `committed_tokens == len(seq) - 1`。
- 前缀快照的 token 边界必须等于共享 KV 前缀边界。
- 预留失败只撤销增量；前向失败会终止当前引擎的后续执行。
- 请求完成释放 KV 与 Slot；缓存发布与淘汰各自只增加或释放自己持有的引用。

对应测试见 [test_scheduler.py](../tests/unit/test_scheduler.py)、[test_prefix_cache.py](../tests/unit/test_prefix_cache.py)、[test_state_manager.py](../tests/unit/test_state_manager.py) 和 [test_llm_engine.py](../tests/unit/test_llm_engine.py)。
