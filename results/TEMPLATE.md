# <日期> — <模型> — <设备> — <实验>

状态：PENDING / PASS / FAIL / BLOCKED

## 环境

| 字段 | 实际值 |
| --- | --- |
| 日期 / 执行人 / Git commit / dirty | |
| 模型 / 检查点 revision / dtype | |
| GPU / 显存 / 驱动 / CUDA | |
| Python / PyTorch / Triton | |
| Transformers / FlashInfer | |
| KV dtype / K、V scale / prefix 配置 | |

## 复现输入

- 实验类别：HF Parity / Prefix / Baseline / Profile / Multimodal / Graph。
- 完整命令、prompt / token ID、随机种子与生成参数：
- Batch / token budget / max_model_len / KV block / Graph：
- 预热轮数、测量轮数、重复次数、跳过数：
- 原始输出 JSON / 日志 / Chrome Trace 路径：

## 正确性

- 逐 token 对齐结论、首个不一致的位置和双方 token：
- logits 或逐层误差、阈值：
- Prefix snapshot 边界与大小；是否独立证明状态恢复：
- 是否确认 CUDA Graph 实际 replay：
- 未覆盖的工作负载：

## 性能（仅实测后填写）

| 指标 | 单位 | 实测值 |
| --- | --- | --- |
| 请求、输出与总 token 吞吐 | req/s、tok/s | |
| Prefill / Decode 执行吞吐 | tok/s | |
| TTFT P50/P99 | ms | |
| TPOT P50/P99 | ms | |
| E2E P50/P99 | ms | |
| CUDA 峰值已分配显存 | GiB | |
| Attention / GDN / MoE 的时间与占比 | ms、% | |

## 结论

- 失败与环境阻塞原因、测量波动、适用边界：
- 候选优化、修复提交和同负载复测：

口径见[验证文档](../docs/validation.md)。不以编写了命令或结果模板代表实验已完成。
