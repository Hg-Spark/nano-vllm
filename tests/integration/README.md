# 真实权重集成验证

[测试导航](../README.md) · [完整验证讲解](../../docs/03_validation_performance.md)

本目录保存验证说明，当前没有由 pytest 自动执行的真实权重测试。集成验证通过仓库 `scripts/` 下的入口手动运行，需要可用的 CUDA 环境、本地检查点和足够容纳模型的显存。

## 准备

从仓库根目录执行，先完成[快速开始](../../docs/00_quickstart.md)，设置模型路径：

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
```

模型必须属于 Qwen3.5-MoE，权重非量化。验证脚本使用 Transformers 的 `AutoModelForCausalLM` 作为 text-only HF 参考加载入口，与本分支的实现范围一致。

## 验证顺序

```bash
# 1. HF 与本实现的逐 token 贪心比较，再检查 BF16 联合前缀恢复
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" \
  --max-new-tokens 32 \
  --check-prefix-resume

# 2. 对齐失败时，在相同提示上定位逐层误差
python scripts/diagnose_qwen3_5_moe.py "$NANOVLLM_MODEL"

# 3. 正确性可信后测量合成请求负载
python scripts/bench.py "$NANOVLLM_MODEL" --json

# 4. 采集 decode 热点
python scripts/profile_decode.py "$NANOVLLM_MODEL" \
  --context-len 4096 \
  --batch-size 8
```

上述第 1、2 步脚本的默认 prompt 不同。排查具体失败时，必须分别传入相同的 `--prompt`；逐层诊断只检查 prompt 前向，不能覆盖所有 decode/恢复偏差。

## 如何记录结论

- 贪心与前缀恢复结果写入 [parity.md](../../results/parity.md)，保留两类结果及日志。
- 性能与环境写入 [基线模板](../../results/5090_baseline.md)，明确实际设备及是否能装入完整模型。
- 热点和优化依据写入 [profile.md](../../results/profile.md)，保留 trace。
- 无法加载、依赖不匹配、数值失败分别记录；没有运行的实验保持“待验证”。

`verify` 和 `bench` 当前未提供 FP8 dtype/scale 参数，不能把这些命令的通过结果用于证明 FP8 路径。相关接口限制见[验证与性能分析](../../docs/03_validation_performance.md)。


## FlashInfer GPU 集成测试

在 CUDA 13.0 环境中可单独验证真实分页 Attention kernel：

```bash
python -m pytest tests/integration/test_flashinfer_attention.py -q
```

该测试使用 Qwen3.5-MoE 的 Full Attention 几何（16 个 Q heads、2 个 KV heads、head_dim 256），比较 FlashInfer paged prefill/decode 与 PyTorch reference；SM120 还会覆盖 FP8 KV scale 路径。
