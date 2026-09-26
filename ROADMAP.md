# 路线图：先验证，再按测量优化

[项目首页](README.md) · [文档导航](docs/README.md) · [验证流程](docs/03_validation_performance.md)

当前阶段保持功能范围稳定，优先建立可信的正确性与性能证据。下列“已有实现”只表示代码存在；真实权重对齐和性能记录仍应以 [results/](results/README.md) 为准。

## 已有实现

- Qwen3.5-MoE 纯文本、单卡、eager 模型适配器。
- Full Attention 分页 KV 与 GDN 卷积/循环状态。
- 变长打包、连续批处理、带状态的分块 prefill。
- decode 优先的共享 token 预算与运行队列轮转。
- KV/GDN 联合抢占、前向失败后的状态清理与重算准备。
- 带引用计数的联合前缀缓存与 BF16 CPU GDN 快照。
- 可选 FP8 E4M3 KV 存储，分页读取由 FlashInfer 直接执行。
- 单元测试、真实权重验证、逐层诊断、基准与 decode profile 脚本。

## 阶段 0：确认实验能运行

先核对依赖、检查点和完整权重显存。激活专家规模不能代替权重规模；当前没有量化权重、CPU 卸载或多卡路径。

验收：模型能够完整加载，单请求前向能够执行，实际软件版本和设备容量已记录。若目标设备放不下权重，记录容量阻塞，不生成虚假的吞吐基线。

## 阶段 1：确认数值可信

执行 [真实权重贪心对齐与前缀恢复](docs/03_validation_performance.md)，覆盖多个提示与长度，记录到 [parity.md](results/parity.md)。不一致时先做逐层或针对性续算诊断。

验收：测试输入、生成长度、首个不一致位置、原始日志和结论齐全；贪心输出与 BF16 前缀恢复分别报告结果。

此阶段失败时，优先修复数值或状态语义，不用 kernel 加速掩盖问题。

## 阶段 2：建立可复现基线

在足够显存的设备上运行基准，记录模型 revision、代码提交、软件版本、负载形状、缓存设置与重复次数。现有 [5090_baseline.md](results/5090_baseline.md) 是原定目标的待验证模板；若更换设备，应新增对应记录并明确差异。

验收：TTFT、TPOT、端到端 P50/P99、执行吞吐、输出吞吐和峰值显存有原始数据支持。比较前后结果时保持模型、输入、采样、并发和计时口径一致。

## 阶段 3：定位主要开销

运行 [profile_decode.py](scripts/profile_decode.py)，将测量写入 [profile.md](results/profile.md)。同时检查 trace、实际活动请求数和命名范围是否有效。

| 观测结果 | 候选工作 | 必须重新检查 |
| --- | --- | --- |
| GDN 占主导 | 单步循环或分块 GDN kernel | 状态连续性、chunk/decode 数值 |
| MoE 占主导 | token 分组、分组 GEMM 或融合 dispatch | 路由、权重归一化、共享专家输出 |
| Full Attention 占比显著 | 直接分页读取的 GQA/FP8 kernel | 页面寻址、缩放、数值与总 TPOT |
| 算子已稳定，启动开销仍显著 | 固定批次存储与 CUDA Graph | 捕获安全性、不同 batch/长度和资源回收 |

验收：优化对象对应可复现的主要开销，改动后数值验证和端到端基准都已重跑。局部 kernel 更快不等于阶段通过。

## 暂缓范围

多模态打包、三轴 mRoPE、TP/EP、推测解码、prefill/decode 分离部署、通用后端注册体系、专用状态换出池以及大规模 radix 前缀缓存，都需要具体实验目标或瓶颈证据才进入当前范围。

新增抽象应对应真实的所有权或生命周期需求。取舍说明见[设计文档](docs/04_vllm_sglang_tradeoffs.md)。
