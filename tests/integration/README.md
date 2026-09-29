# 集成验证入口

[测试索引](../README.md) · [完整验证流程](../../docs/03_validation_performance.md)

真实 Qwen3.5-MoE 检查点的对齐、诊断、benchmark 和 profile 通过仓库 `scripts/` 手动运行；具体参数、判定标准和实验记录要求统一维护在 [验证与性能分析](../../docs/03_validation_performance.md)。

准备本地非量化检查点并设置：

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
```

主要入口：

| 目标 | 命令 |
| --- | --- |
| HF 贪心对齐 + 精确状态前缀恢复 | `python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" --check-prefix-resume` |
| 单图 / Chunked Prefill / Decode Graph 对齐 | `python scripts/verify_qwen3_5_moe_multimodal.py "$NANOVLLM_MODEL" /path/to/image.jpg --decode-graph` |
| 逐层数值诊断 | `python scripts/diagnose_qwen3_5_moe.py "$NANOVLLM_MODEL"` |
| 端到端 benchmark | `python scripts/bench.py "$NANOVLLM_MODEL" --json` |
| Decode profile | `python scripts/profile_decode.py "$NANOVLLM_MODEL" --context-len 4096 --batch-size 8` |

实验输出分别记录到 [parity.md](../../results/parity.md)、对应 baseline 文件和 [profile.md](../../results/profile.md)。未执行的实验保持 PENDING。

## FlashInfer GPU 集成测试

```bash
python -m pytest tests/integration/test_flashinfer_attention.py -q
```

该测试直接比较 FlashInfer paged prefill/decode 与 PyTorch reference；SM120 环境还覆盖 FP8 KV scale 路径。
