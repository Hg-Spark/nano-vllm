# 真实权重数值对齐

[实验导航](README.md) · [验证流程](../docs/03_validation_performance.md)

状态：**待验证（PENDING）**。以下命令与字段是实验模板，不代表已有通过结果。

## 环境

| 项目 | 记录 |
| --- | --- |
| 日期／执行人 | 待填写 |
| 模型目录／检查点 revision | 待填写 |
| 权重 dtype／GPU 与显存 | 待填写 |
| Python／PyTorch／CUDA | 待填写 |
| Transformers／Triton／FlashAttention | 待填写 |
| nano-vLLM 提交／未提交改动 | 待填写 |
| 输入模板与分词设置 | 待填写 |

## HF 与本实现的贪心对齐

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --prompt "请解释自回归推理为什么需要 KV cache。" \
  --max-new-tokens 32 \
  --max-model-len 4096
```

| 记录项 | 结果 |
| --- | --- |
| 完整提示／输入 token 数 | 待填写 |
| 实际比较的输出 token 数 | 待填写 |
| HF token 序列 | 待填写 |
| 本实现 token 序列 | 待填写 |
| 首个不一致位置与两侧 token | 待填写；完全一致时写“无” |
| 原始日志／命令 | 待填写 |
| 判定 | 待验证 |

对不同提示和长度分别记录，不用单条成功外推到所有输入。脚本默认路径未开启 FP8 KV。

## BF16 联合前缀恢复

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --max-new-tokens 32 \
  --check-prefix-resume \
  --prefix-block-size 256
```

| 记录项 | 结果 |
| --- | --- |
| 前置 HF 贪心对齐 | 待验证 |
| 实际命中的前缀边界 | 待填写 |
| GDN 快照大小 | 待填写，MiB |
| 无缓存基线 token | 待填写 |
| 恢复后的 token | 待填写 |
| 首个不一致位置 | 待填写 |
| 原始日志 | 待填写 |
| 恢复验证判定 | 待验证 |

这一步比较本实现的两条路径，检查 BF16 快照舍入是否改变所测输入的贪心续算；不证明所有输入上无误差。

## 失败诊断

- 失败阶段：待填写（加载／prefill／decode／快照／恢复等）。
- 复现命令与相同 `--prompt`：待填写。
- 逐层诊断的 token 上限与相对 RMS 阈值：待填写。
- 首个超过阈值的层／形状／误差：待填写。
- 进一步缩小的算子或状态问题：待填写。
- 修复提交与复测结果：待填写。

逐层诊断入口为 [diagnose_qwen3_5_moe.py](../scripts/diagnose_qwen3_5_moe.py)；该探针只覆盖 prompt 前向，decode 或恢复专有问题需要另行定位。
