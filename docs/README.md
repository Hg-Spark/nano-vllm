# 中文文档导航

[返回项目首页](../README.md)

本文档对应 `feature/qwen35-moe` 分支，初次重构时核对的代码基线为 `fab4db6a32b5adda03df4deb37cefa85f58221f3`。后续阅读时，以当前分支代码、配置和实验记录为准。

## 建议阅读顺序

| 顺序 | 文档 | 读完应能回答的问题 |
| --- | --- | --- |
| 0 | [快速开始与配置](00_quickstart.md) | 怎样运行？哪些参数影响容量与延迟？ |
| 1 | [模型与运行时架构](01_qwen35_hybrid_runtime.md) | 一个 token 如何经过 Attention/GDN 和 MoE？ |
| 2 | [调度与混合状态管理](02_scheduler_cache.md) | 为什么 KV 和 GDN 必须一起提交、释放和恢复？ |
| 3 | [验证与性能分析](03_validation_performance.md) | 哪些证据能支持正确性和性能结论？ |
| 4 | [与 vLLM、SGLang 的设计取舍](04_vllm_sglang_tradeoffs.md) | 为什么保留这些模块，又暂缓那些功能？ |
| 5 | [源码阅读与讲解指南](05_code_walkthrough.md) | 如何沿调用链讲清一个请求的生命周期？ |

只想运行模型，可以先读第 0 章；学习推理引擎，可以按 1 → 2 → 5 的顺序对照代码；准备修改 kernel，应先完成第 3 章的验证步骤。

## 术语速查

| 术语 | 本仓库中的含义 |
| --- | --- |
| token | 分词后的整数编号，不等同于一个汉字或一个英文单词 |
| prefill／预填充 | 处理尚未进入历史状态的一段输入；本分支允许分块完成 |
| decode／解码 | 每个被选中的请求处理上一步生成的一个 token，再预测下一个 |
| continuous batching／连续批处理 | 按调度轮次让请求进入或退出执行集合，不必等待整批同时结束 |
| packed prefill／变长打包 | 拼接多个请求的 token，用偏移量保留请求边界 |
| KV cache | Full Attention 保存的历史 key/value 张量 |
| paged KV／分页 KV | 用固定 token 容量的物理块保存 KV，通过块表查找 |
| GDN | Gated DeltaNet，依赖卷积历史和循环矩阵的序列算子 |
| state slot／状态槽 | 一个活动请求在每个 GDN 层中对应的固定槽位 |
| MoE | 混合专家层，包含被路由选中的专家和共享专家 |
| Top-K | 对每个 token 选择得分最高的 K 个专家；不是生成采样的 top-k 参数 |
| commit／提交 | 将已成功执行的 token 数计入 `committed_tokens` |
| snapshot／快照 | 某个 token 边界处全部 GDN 层的状态副本 |
| preemption／抢占 | 释放一个请求占有的混合历史，之后从可复用前缀恢复或重算 |
| greedy parity／贪心对齐 | 相同输入和确定性生成设置下，输出 token 序列逐项一致 |
| eager | 按当前调用路径执行算子；本分支未使用 CUDA Graph |

## 文档与证据的分工

- `docs/` 解释实现、示例和设计取舍；示意数字会明确标注。
- [tests/](../tests/README.md) 说明测试覆盖范围和运行条件。
- [results/](../results/README.md) 保存可复现的实测记录。“待验证”不能当作通过证明。
- [ROADMAP.md](../ROADMAP.md) 记录下一步工作的条件和验收要求。

修改实现时，应同步更新相应章节中的参数、命令和行为说明。文档中的代码链接指向本仓库文件；外部设计资料仅用于解释借鉴关系。
