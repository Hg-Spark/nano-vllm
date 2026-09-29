# 路线图：先验证，再按测量优化

[项目首页](README.md) · [验证方法](docs/03_validation_performance.md) · [实验记录](results/README.md)

本文件只记录尚未完成的验证、测量和后续工作。当前支持范围以 [README](README.md#支持范围) 为准，具体执行方法统一见 [验证与性能分析](docs/03_validation_performance.md)。

## 阶段 0：确认实验能运行

状态：**PENDING**

验收条件：

- 完整检查点能够加载；
- 单请求前向能够执行；
- 实际 GPU、显存和软件版本已记录；
- 若容量不足，明确记录为环境阻塞，不生成性能结论。

## 阶段 1：确认数值可信

状态：**PENDING**

覆盖真实文本提示、联合前缀恢复以及单图路径，并将结果写入 [parity.md](results/parity.md)。

验收条件：输入、生成长度、首个不一致位置、原始日志和结论齐全；失败时先修复数值或状态语义。

## 阶段 2：建立可复现基线

状态：**PENDING**

在足够显存的设备上记录模型 revision、代码提交、负载、缓存设置、重复次数和完整指标。原定 RTX 5090 模板见 [5090_baseline.md](results/5090_baseline.md)；若设备不满足容量要求，应新增对应设备记录。

验收条件：TTFT、TPOT、E2E P50/P99、执行/输出吞吐和峰值显存均有原始数据支持。

## 阶段 3：按 Profile 选择优化对象

状态：**PENDING**

热点记录写入 [profile.md](results/profile.md)。只有可复现的主要开销才进入优化：

| 观测 | 候选方向 |
| --- | --- |
| GDN 占主导 | 单步循环或分块 GDN kernel |
| MoE 占主导 | token 分组、Grouped GEMM 或 dispatch 融合 |
| Attention 占比显著 | 更深的 GQA / Paged / FP8 kernel 优化 |
| kernel 已稳定、启动开销显著 | CUDA Graph bucket / capture 策略 |

验收条件：优化后重新执行对应数值验证和同负载端到端 benchmark；局部 kernel 加速不能单独视为完成。

## 暂缓范围

多图/视频、TP/EP、推测解码、Prefill/Decode 分离部署、通用后端注册体系、专用状态换出池、大规模 radix 前缀缓存，以及没有测量依据的新 kernel。

设计取舍与引入新抽象的判断标准见 [docs/04_vllm_sglang_tradeoffs.md](docs/04_vllm_sglang_tradeoffs.md)。
