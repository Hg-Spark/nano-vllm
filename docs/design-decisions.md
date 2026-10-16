# 与 vLLM、SGLang 的对比和技术选型

本章解释为什么以 nano-vLLM 为基线实现 Qwen3.5 Dense/MoE Hybrid Runtime、哪些设计参考了 vLLM/SGLang，以及哪些成熟机制暂时不引入。比较对象为 **2026-10-13 核对的官方文档/设计资料**，具体行为取决于各框架的版本、模型与后端。本章没有跨框架性能排名；真实模型的验证范围见[验证文档](validation.md)。

## 一、为什么选择 nano-vLLM 作为基线

项目的目标是理解、实现并验证以下完整链路：

```text
Qwen3.5 Hybrid Model → Full Attention KV + GDN Conv/Recurrent State
  → 连续批处理 / Chunked Prefill / Decode-first
  → 请求级资源预留、逻辑提交、抢占与 Prefix 恢复
  → HF 数值对齐、端到端测量、Profile 驱动优化
```

选择起点时的评价标准是：①能否直接定位/修改调度与状态所有权；②能否为每条路径构造独立测试；③为了实现 Qwen3.5 必须理解和维护多少无关组件；④目标是否需要生产级服务能力。

| 方案 | 直接收益 | 必须承担的成本 | 此项目的选择 |
| --- | --- | --- | --- |
| 扩展 vLLM V1 | 已有多模型、KV 缓存、调度、并行和服务设施；适合最终部署 | 需要遵循缓存组、ModelRunner、Attention 后端与较大框架的接口及版本演进 | 不作为首次实现的宿主；借鉴调度与 Hybrid KV 的约束 |
| 扩展 SGLang | RadixAttention、缓存感知调度、HiCache、多级复用和服务基础设施 | 要处理 RadixTree/状态缓存与 Scheduler、内存池、后端之间的协同 | 不把完整缓存/调度栈引入教学与验证运行时 |
| 基于 nano-vLLM 精简扩展 | 直接控制 Sequence、Scheduler、ModelRunner、Paged KV 与失败路径；源码链路短 | 自行适配模型、验证状态语义；没有现成的工业级能力与性能保证 | **当前选择**：限定单 GPU、Qwen3.5 Dense/MoE，先验证再优化 |

此选择与开源引擎优劣无关：**生产部署优先评估 vLLM/SGLang**，而本项目优先使“两个物理状态、一个逻辑提交边界”可解释、可测试。保留小运行时的代价包含额外正确性验证、功能覆盖不足，以及可能更高的调度与 kernel 开销。

## 二、逐模块比较：参考了什么，为什么这样实现

| 问题 | vLLM V1 的方案 | SGLang 的方案 | 本项目与选型原因 |
| --- | --- | --- | --- |
| 请求调度 | RUNNING/WAITING、token budget、Chunked Prefill；运行请求优先，具体策略由配置控制 | PrefillAdder、缓存感知/其他队列策略、Chunked Prefill，支持可选 Mixed Chunk | `Scheduler.schedule()` 先 Decode、再按余量安排 Prefill；无需为动态策略增加更多状态 |
| Prefill / Decode 执行 | V1 Scheduler 可以在一轮分配两类请求的 token；执行后端按模型和模式处理混合批次 | 可启用 Mixed Chunk，把部分 Prefill 与 Decode 合入执行批次 | 一轮调度、**两次 forward**（Decode → Prefill）；牺牲部分融合机会，简化 GDN Slot 与 Attention 元数据 |
| 显存定容 | 默认借助模型峰值 Profiling 推导 KV 空间，并支持显式配置 KV 容量 | `mem_fraction_static` 控制模型权重与 KV Pool 等常驻预算，配合 Chunk/并发调节 | 不做启动 Profiling，按 `gpu_memory_utilization` 静态扣减已用显存、GDN/Vision 预留；减少专用执行路径，但需负载测试校准余量 |
| 混合状态内存 | Hybrid KV 按层类型建立不同 cache spec / manager，并协调各组缓存有效性 | Unified Radix Cache 将 FULL/SWA/循环态组件放入统一前缀结构（以官方设计材料为准） | `BlockManager` 管分页 KV，`StateSlotManager` 管 GDN Slot，`Scheduler` 完成联合预留与释放；不引入通用 cache-spec 注册体系 |
| 共享前缀 | Automatic Prefix Caching 以缓存块为基础；Hybrid 场景还要协调各组可用前缀 | RadixAttention / RadixTree 表示 token 前缀，HiCache 扩展到 GPU/CPU/外部存储 | 精确 token 前缀 + 有界 LRU，KV 引用共享、GDN CPU Snapshot 绑定同一边界；目标是验证复用正确性 |
| 循环状态恢复 | Hybrid 的 Mamba/循环缓存需满足对应的 checkpoint、对齐与保留规则 | 循环态在安全前缀边界需要 checkpoint，多个组件共同决定可恢复边界 | 仅完整 KV 页边界捕获 GDN 状态；KV 命中但缺少对应 GDN 快照时**不能单独复用 KV** |
| 抢占与故障 | 生产引擎具有更完整的资源回收、请求重排及扩展缓存/通信路径 | 与 Cache/Allocator、请求生命周期和缓存复用协同 | 资源不够则淘汰/抢占、联合释放并重算；执行失败丢弃受影响的物理历史且异常上抛 |
| Full Attention | 支持多种 Attention 后端及针对模型/硬件的优化 | 支持多后端，也可按阶段组合后端 | 固定 FlashInfer Paged Prefill/Decode 与 KV Append；FP8 缩放及转换使用 PyTorch，避免自行维护写入 kernel |
| GDN / MoE | 对不同架构集成专用层、backend、融合/编译等优化 | 具备多种推理 kernel 及 MoE/混合 Attention 后端 | GDN 在 SM90/100/120 且 K/V head_dim=128 时使用 FlashInfer，其他配置使用 PyTorch FP32 chunk=64；MoE Prefill/Decode 使用 FlashInfer CUTLASS Fused MoE；先建立明确的对照基线 |
| CUDA Graph | 有后端能力协商与多种 Graph 捕获模式 | 与执行批次及 Attention Backend 的 Graph 支持共同设计 | 只为显式指定的 Decode Batch size 做 exact bucket；不做 padding，未命中走 Eager |
| 多模态与并行 | 成熟的模型适配、服务/并行能力 | 成熟的模型适配、服务/并行能力 | 只支持单图、单 GPU，暂缓多图/视频及 TP/EP；保持请求状态问题的实验范围明确 |

上表概括机制，不把 vLLM 的 Mamba 与 Qwen3.5 GDN 当作相同算子，也不声称 SGLang 的任意版本默认启用 Unified Radix Cache、Mixed Chunk 或 HiCache。官方依据见文末。

## 三、四个核心设计取舍

### 1. KV 和 GDN 为什么物理分开，却必须共同提交？

Attention 的历史是随长度增长、可共享的 KV 页；GDN 包含固定槽中的卷积状态与循环矩阵。两者容量、所有权、可复用方式不同。如果仅按 KV 命中长度继续 Decode，而 GDN 保留另一段历史，模型在相同 token 上会读取两个不一致的上下文。

本项目用 `Sequence.committed_tokens` 作为共同逻辑边界：`KV有效前缀 = GDN有效前缀 = committed_tokens`。分配和释放则分别委托 `BlockManager` 与 `StateSlotManager`，由 `Scheduler` 集中执行联合操作。vLLM 的 Hybrid KV 按不同层组管理可复用缓存，SGLang Unified Radix Cache 允许组件对安全前缀边界各自检查；本项目将问题收敛到 **Full Attention + GDN 两种状态**，避免提前建设泛化的多类型缓存框架。

对应代码：[sequence.py](../nanovllm/engine/sequence.py)、[scheduler.py](../nanovllm/engine/scheduler.py)、[cache_runtime.py](../nanovllm/engine/cache_runtime.py)，完整不变量见 [调度与状态管理](scheduling.md)。

### 2. 为什么没有直接采用 RadixTree 或 HiCache？

SGLang RadixAttention 适合大量请求反复共享前缀的场景；HiCache 可进一步跨 GPU、主机和外部存储扩大缓存，但引入匹配、数据搬运、预取与淘汰的复杂度。当前缓存默认关闭，开启后条目有界。现阶段更需要检查：

1. 命中的 token、整块 KV 和 GDN Snapshot 是否对应同一边界；
2. 请求、缓存条目各自持有的 KV 引用是否正确增减；
3. 恢复后的生成是否与不命中缓存一致，且真实恢复路径确实执行。

例如，块容量为 16，请求 A 已处理 20 个 token，GDN 状态在位置 20；即便 KV 的第一个完整块对应前 16 个 token，也不能直接把位置 20 的 GDN 状态当作位置 16 的 Snapshot。必须在位置 16 **实际捕获过**循环状态，才能发布对应的联合前缀条目。

因此先使用 **整块边界、精确匹配、有界 LRU、独立快照**。它的局限是扫描成本、快照主机内存与 D2H/H2D 数据移动；只有测得高命中率、条目数量或传输成为瓶颈，再评估 radix、分层缓存或更稀疏的状态 checkpoint。

### 3. 为什么 Prefill / Decode 不合并成一次 Forward？

三种实现都希望减小长 Prompt 对生成延迟的影响。vLLM 的 Chunked Prefill 优先 Decode，再安排剩余预算；SGLang 有可选 Mixed Chunk 路径。当前 Scheduler 也允许一轮同时安排两类 token，但 `LLMEngine.step()` **先执行 Decode Batch 并提交，再执行 Prefill Batch 并提交**。

分开的收益：

- Decode 始终只有每请求一个新 token；GDN Decode 可以按 Slot 向量化，Prefill 则要处理变长分段、非零历史和可选视觉区间。
- FlashInfer 的两个 Attention wrapper 各自规划元数据，不需要一次 forward 内动态选择混合布局。
- 失败边界清楚：Decode 成功后 Prefill 失败，前一批结果与历史不会被整体回滚。

成本是可能多一次模型调度、kernel launch 与相关开销。待 Profile 证明这些成本显著、且有混合 GDN 正确性测试覆盖后，才考虑合并 Forward；不能因为其他框架存在混合执行路径就认定本项目合并后一定更快。

### 4. 为什么删除启动 Profiling Warmup，却仍保留 CUDA Graph 预执行？

当前 [ModelRunner](../nanovllm/engine/model_runner.py) 在加载权重并建立 MoE/FlashInfer Workspace 后，由 [cache_runtime.py](../nanovllm/engine/cache_runtime.py)按下式计算物理 KV Block 数，不执行无持久 KV 的 SDPA 或临时 GDN 路径：

缓存预算公式与预留口径统一见[运行时缓存分配](runtime.md#启动时怎样分配缓存)。


默认 `gpu_memory_utilization=0.8`。`GDN_slot_reserve` 按 `max_num_seqs` 个活动槽计入，视觉特征预留按当前单图范围与最大上下文估算；物理 KV Pool 仍一次性分配，Scheduler 只按需分配其逻辑 Block。这里预留的显存供临时激活、内部 Workspace 和 Graph Capture 使用，**并非测得的峰值上界**。

选择该方式主要是删除启动专用的 `is_warmup` 状态、Attention SDPA Forward 和临时 GDN State 分支，保持首次及续接 Prefill 始终走同一套 FlashInfer Paged KV/持久 State Slot。代价是峰值负载可能 OOM、首请求可能承担 Kernel 编译、静态比例可能留下过多未用显存；没有真实 GPU 边界测试前，不能声称它比 Profiling 更高效或更安全。

这与两个框架的侧重点不同：[vLLM V1 的显存探测](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)主要测量模型峰值再定容，[SGLang 的显存调优](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)将静态比例、Chunk Size、最大并发与部署压力测试配合使用。**本实现借鉴静态预算的简单性，不宣称它们没有其他启动预执行或 Profiling。**

当显式开启 Decode CUDA Graph 时，`DecodeGraphBucket.capture()` 仍需真实预执行模型来初始化 Kernel、随后捕获固定地址图，并清理用于捕获的 KV/GDN 状态；这是 Graph Capture 正确性所需流程，不能与已经删除的显存容量 Profiling 混为一谈。

## 四、其余取舍及改变条件

| 当前方案 | 有意承担的代价 | 什么证据会推动重新选型 |
| --- | --- | --- |
| 静态 KV 显存预算（默认 0.8） | 可能保留过量余量或在峰值负载时 OOM | 按最大 Prefill、最大 Decode Batch、单图负载实测，才评估增减比例或恢复显存 Profiling |
| 抢占后联合释放并重算 | 长上下文重复 Prefill | 测得重算成本显著，才评估请求状态 swap/checkpoint |
| 前缀 GDN Snapshot 保留 FP32 循环矩阵 | CPU 内存及拷贝开销较大 | 命中率、快照带宽与内存测量说明需要压缩；有精度验证再引入有损模式 |
| FlashInfer Paged Attention + FP8 KV 可选 | 受 FlashInfer 和 GPU 架构能力约束 | Kernel trace 和端到端测量显示现有 Attention 路径是主要瓶颈 |
| 独立 PyTorch GDN Chunk | 存在中间矩阵和计算开销 | 数值对齐后，Profile 显示 Prefill GDN 是主要热点 |
| FlashInfer CUTLASS MoE | GPU 架构覆盖有限（SM75 不支持），无内部替代后端 | 明确需要更多 GPU 架构、且有可维护的正确性/性能基线 |
| Decode CUDA Graph exact bucket | Batch 未命中时回退 Eager；每个 bucket 消耗显存 | 测得更多 bucket 或 padding 在具体负载下稳定改善 TPOT |
| 单 GPU / 单图 | 无 TP/EP、跨节点缓存、多图视频服务 | 项目目标转向生产服务或更大模型部署 |

**验证优先级：** HF Greedy 对齐 → Chunked Prefill 连续性 → Prefix 实际恢复 → 失败/抢占后的数值一致性 → 相同设备/模型/输入的 benchmark → Profile。已有 Dense FP16 实测仅证明记录范围；[验证文档](validation.md)明确区分未运行和通过。

## 五、何时改用 vLLM / SGLang，何时继续保留小运行时

- **转向生产部署**：需要高并发服务、扩展并行、更多模型/量化、长上下文与观测治理时，优先在 vLLM/SGLang 上验证需求；重复自建这些设施会扩大维护面。
- **继续 nano-vLLM 专项研究**：目标仍是解释和验证 Qwen3.5 Hybrid 的状态语义、Scheduler 与 Kernel 优化因果时，保持专用执行路径能减少干扰变量。
- **引入新的抽象**：同时满足存在独立的资源生命周期/失败语义、已有重复代码，以及能够用测试证明抽象降低复杂度。仅为了“对齐大型框架”不引入接口。

本项目未完成三者同模型、同硬件、同输入的公平性能比较，因此选型理由只基于任务范围、机制与工程成本，不使用无依据的吞吐或延迟优劣判断。暂缓功能与下一阶段验收见 [ROADMAP](../ROADMAP.md)。

## 官方参考（原有参考 2026-10-13 核对，显存资料于 2026-10-14 复核）

1. [vLLM — Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)：按层类型建立缓存规则及前缀安全边界；该设计页注明基于特定提交，不能替代当前所有版本的 API。
2. [vLLM — Scheduler 实现](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/) 与 [Chunked Prefill 调优](https://docs.vllm.ai/en/stable/configuration/optimization/)：token budget、RUNNING/WAITING、chunked prefill。
3. [vLLM — CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)：后端能力与多种捕获策略。
4. [SGLang — HiCache Design](https://docs.sglang.io/docs/advanced_features/hicache_design)：RadixAttention、HiRadixTree、GPU/CPU/外部存储层级。
5. [SGLang — Unified Radix Cache](https://www.sglang.io/blog/unified-radix-cache)：FULL/SWA/MAMBA 组件、安全前缀边界与 checkpoint；具体部署版本和默认行为需另行核实。
6. [SGLang — 调度实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) 与 [Schedule Policy](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)：请求调度、批次构造与缓存感知策略，比较时应固定版本。
7. [vLLM V1 — GPUWorker](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu_worker/)：`determine_available_memory()` 通过模型峰值 Profiling 定容，显式 KV 容量配置下仍可能需要执行编译用的 Profile Run。
8. [SGLang — Hyperparameter Tuning](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)：`mem_fraction_static`、Chunk Size、最大并发与 OOM 调参边界。
