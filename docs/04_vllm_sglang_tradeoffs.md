# 与 vLLM、SGLang 的设计取舍

[文档导航](README.md) · 上一章：[验证与性能](03_validation_performance.md) · 下一章：[源码阅读与讲解](05_code_walkthrough.md)

本章解释本分支为什么采用当前结构。比较标准是“能否清楚地实现和验证 Qwen3.5-MoE 的混合状态生命周期”。它不是三个项目的性能排名，也不表示本分支具备生产引擎的完整服务能力。

## 1. 外部设计给出的启发

vLLM 的混合缓存设计区分不同层类型的容量与前缀命中规则，并讨论状态大小、页面大小和缓存分组之间的约束。由此可以得到一个可迁移的原则：同一请求的不同历史表示需要协调，但不必具有相同物理布局。[vLLM：Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

SGLang 的 HiCache 用 GPU、主机内存和存储后端组织多级缓存，以 HiRadixTree 记录前缀及其数据位置，并协调匹配、预取和写回。它展示了扩大缓存规模时需要承担的数据移动与元数据管理成本。[SGLang：HiCache System Design and Optimization](https://docs.sglang.io/docs/advanced_features/hicache_design)

以上是对官方设计资料的概括，核对日期为 2026-09-22。vLLM 页面注明其设计基于特定提交；不能把某份设计文档中的阶段性描述当作所有最新版本的功能结论。下面关于本仓库的选择，是结合[本地实现](01_qwen35_hybrid_runtime.md)作出的工程解释，不是外部项目的承诺。

## 2. 为什么保留一个专用运行时

| 实现方向 | 对本项目的价值 | 需要承担的代价 |
| --- | --- | --- |
| 在生产引擎中扩展 | 能利用已有服务、缓存、并行和 kernel 体系 | 要先理解更大的执行框架，局部不变量更难单独展示 |
| 保持本分支专用实现 | 调度、所有权、状态恢复和模型计算可以逐层追踪 | 要自行验证正确性，当前缺少大量优化和服务设施 |

本分支选择后者：让“分页 KV + GDN 状态 + MoE 路由 + 一个提交边界”可以被直接检查。若目标变成部署真实服务，应该重新评估基础设施、模型支持、运维和性能需求，而不能仅凭代码较少决定选型。

## 3. 各项选择的收益、成本与改变条件

| 选择 | 当前收益 | 当前成本 | 什么证据值得推动改变 |
| --- | --- | --- | --- |
| 一个 decode 优先预算 | token 计数、轮转和回滚规则直观 | prefill 可能被延后，缺少自适应流量策略 | 固定负载下的 TTFT/尾延迟证明准入策略不足 |
| KV 与 GDN 分开管理 | 共享块和独占槽的所有权清楚 | 恢复时需要显式协调两类资源 | 出现真正不同的新资源生命周期后再提取接口 |
| 单一 `committed_tokens` | 两种历史有共同恢复边界 | 失败时不能任意保留某一侧较新的进度 | 必须有完整的一致性协议才能放宽 |
| 增量 KV + 持续占用 GDN 槽 | 长输入分轮计算，避免入队就预留全长 KV | 分块请求仍占状态槽和已分配 KV | 容量与延迟测量推动调整 chunk 策略 |
| 联合释放后重算 | 抢占和前向失败恢复路径容易验证 | 长请求重算可能昂贵 | 测得重算成本显著后评估 checkpoint/swap |
| 精确前缀匹配 + 有界 LRU | 命中和引用计数容易检查 | 查找扫描条目，缓存规模小 | 查找规模成为瓶颈后再评估 radix 结构 |
| 保精度 CPU GDN 快照 | 释放活动槽后仍能复用，恢复不额外量化循环状态 | FP32 循环状态增加主机空间与设备到主机传输成本 | 命中率和传输开销证明需要更激进压缩后再引入有损模式 |
| FlashInfer GDN Prefill / fused MoE Decode + 参考 fallback | 支持设备走融合快路径，同时保留可读数值基线 | 受 FlashInfer 架构覆盖、workspace 与实际 workload 影响 | 实机 profile 证明收益不足或出现热点后再定制 kernel |
| FlashInfer 分页 Attention + FP8 KV | 统一 prefill/decode 分页读取，避免自维护 Attention kernel | 依赖 FlashInfer 版本、SM120 kernel 覆盖与实际 workload | 固定版本完成数值与 profile 后再决定是否定制 kernel |
| prefill/decode 两次 forward | 两种批次元数据较简单 | 重复遍历模型，存在调度和启动成本 | 主要算子优化后，剩余开销值得合并执行 |
| 显式固定 batch Decode CUDA Graph | 固定地址元数据清晰，不改变 Scheduler；未命中 bucket 直接走 Eager | 每个 bucket 有 workspace/graph pool 成本，当前不做 padding | 实机 profile 证明更多 bucket 或 padding 能稳定改善 TPOT 后再扩展 |

## 4. 三个容易混淆的设计问题

### 物理分开，为什么逻辑还要统一

KV 块按历史长度增长并允许共享；GDN 槽是固定大小的活动请求状态；GDN 快照又是可复用的状态副本。三者的释放规则不同，但继续生成时必须对应同一前缀。

因此，代码共享的是提交边界，而不是把所有资源强行包装成同一种缓存。具体恢复过程见[调度与混合状态](02_scheduler_cache.md)。

### 有前缀缓存，为什么没有 radix tree

当前默认关闭联合前缀缓存；显式设置 `max_prefix_cache_entries>0` 才启用。启用后仍采用小规模精确扫描，使最长前缀匹配、LRU 和独立 KV 引用都易于理解。如果真正昂贵的是 GDN 快照传输，先换查找数据结构并不能直接消除传输成本。

先测量命中率、快照大小、传输耗时、KV 占用和查找成本，再确定优化对象。

### 为什么 CUDA Graph 只做 exact bucket

vLLM 和 SGLang 的成熟运行时会维护更通用的 capture shape、padding、静态 buffer 和后端元数据。本分支只为显式配置的 batch size 捕获固定地址 Decode Graph，例如 `(1, 2, 4)`。B=3 没有对应 bucket 时直接走 Eager，不为了凑 B=4 构造 dummy KV page、dummy GDN slot 或 dummy sampling row。

这样 Graph 仍是 ModelRunner 的执行优化，而不是 Scheduler 的新语义。capture OOM 时保留已经成功的小 bucket、释放当前及更大 bucket；replay 失败则上抛，让现有 failed-step recovery 丢弃可能已经写过的 KV/GDN 物理状态。

### 为什么不先写一个 Attention kernel

FlashInfer 已承担 Full Attention 分页读取，并为 GDN Prefill 和 Decode MoE 提供当前快路径；参考 fallback 仍保留。是否继续优化 Attention、GDN 或 MoE，应由目标设备上的 profile 决定。

应通过 [profile_decode.py](../scripts/profile_decode.py)和端到端基准确定方向。某个 kernel 变快还需要观察是否改善总吞吐或 TPOT，且必须重新进行数值验证。

## 5. 何时引入新抽象

具体暂缓范围统一维护在 [ROADMAP](../ROADMAP.md#暂缓范围)，避免在设计文档中重复维护功能清单。

引入新抽象前检查三个问题：

1. 是否对应已经存在的能力或可测瓶颈；
2. 是否管理独立的所有权、资源寿命或失败恢复规则；
3. 是否让实际执行路径更容易验证和解释。

下一章提供[源码阅读路线与中文讲解提纲](05_code_walkthrough.md)，将这些取舍对应到具体类和状态变化。
