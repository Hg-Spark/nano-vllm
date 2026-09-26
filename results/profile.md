# Decode 热点分析

[实验导航](README.md) · [测量解释](../docs/03_validation_performance.md)

状态：**待验证（PENDING）**。尚无热点占比或优化收益的实测结论。

## 环境与设置

- 日期／执行人：待填写。
- 模型与检查点 revision：待填写。
- 实际 GPU／显存：待填写。
- nano-vLLM 提交／未提交改动：待填写。
- Python／PyTorch／CUDA／Triton／Transformers／FlashInfer：待填写。
- KV dtype／K scale／V scale：待填写。
- 配置 batch／测量窗口实际活动请求数：待填写。
- context／token ID／token 预算：待填写。
- 预热轮数／采集轮数／重复次数：待填写。

## 待执行命令

```bash
python scripts/profile_decode.py "$NANOVLLM_MODEL" \
  --context-len 4096 \
  --batch-size 8 \
  --max-batched-tokens 2048 \
  --warmup-steps 4 \
  --profile-steps 16 \
  --kv-cache-dtype auto \
  --attention-threshold 0.15 \
  --trace results/decode-trace.json
```

FP8 对照可使用 `--kv-cache-dtype fp8_e4m3`，但该脚本未暴露 scale 选项，会使用默认 `1.0`。它提供性能观测，不提供精度通过证明。

在容量允许时逐步测试 batch `1/2/4/8/16`、context `1K/4K/8K/16K` 与两种 KV dtype。不要把候选矩阵视作已验证容量。

## 测量结果

| 配置 batch／实际活动数 | context | KV dtype | 窗口 ms | Attention % | GDN % | MoE % | 三者中的最大项 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |

- KV 写入时间：待填写，ms。
- 原始输出与 trace 路径：待填写。
- 是否出现 prefill／抢占／请求完成：待填写。
- 重复测量波动：待填写。

占比以命名范围的设备时间除以 CUDA event 窗口时间。Attention 范围不含独立的 KV 写入；三项之和不要求等于 100%。若全为零，先检查 profiler 与 trace，不能把脚本选出的最大项视为有效热点。

提示由同一 token 重复构成，专家路由分布可能不代表真实文本。优化后应补充真实提示下的端到端测量。

## 优化决策

- 数值验证记录：待填写。
- 主要耗时及证据：待填写。
- 选择的改动与预期减少的开销：待填写。
- 暂缓其他方案的理由：待填写。
- 改动后的数值复测：待填写。
- 同负载端到端基准：待填写。
- 结论：待验证。

默认 attention 阈值 15% 只用于挑选值得尝试的方案；超过阈值不表示已经实现加速。
