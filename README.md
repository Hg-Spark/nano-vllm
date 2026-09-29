# Nano-vLLM：Qwen3.5-MoE 混合推理运行时

这个分支专注于 **Qwen3.5-MoE 单卡混合推理**。运行时同时管理 Full Attention 的分页 KV 历史与 Gated DeltaNet 的循环状态，并支持文本/单图请求、Continuous Batching、State-aware Chunked Prefill、联合抢占与前缀恢复，以及可选 Decode CUDA Graph。

```text
Full Attention  → 分页 K/V，随上下文增长
Gated DeltaNet  → Conv State + Recurrent State，每个活动请求独占槽位
Sparse MoE      → 每个 token 路由到 Top-K 专家并叠加共享专家
```

GDN Prefill 与 Decode MoE 在支持设备上使用 FlashInfer 快路径，并保留参考路径用于回退和数值验证。真实检查点对齐与性能结果仍以 [results/](results/README.md) 中的实验记录为准。

完整文档阅读顺序见 [docs/README.md](docs/README.md)；安装和运行见 [快速开始](docs/00_quickstart.md)，验证方法见 [验证与性能分析](docs/03_validation_performance.md)，后续工作见 [ROADMAP.md](ROADMAP.md)。

## 支持范围

| 方面 | 当前实现 |
| --- | --- |
| 模型 | Qwen3.5-MoE：语言模型、视觉塔、单图输入 |
| 执行 | 非量化 safetensors；单 GPU；Prefill Eager；Decode 可选固定 batch CUDA Graph |
| 模型层 | FlashInfer 分页 Attention、三轴交错 mRoPE、Gated DeltaNet、Top-K MoE 与共享专家 |
| 调度 | 变长 Packed Prefill、Continuous Batching、State-aware Chunked Prefill、Decode-first token budget |
| 状态 | Paged KV + GDN State Slot；联合抢占、失败恢复、可选联合前缀缓存 |
| 多模态 | 单图 Processor 展开、跨 Chunk 视觉回填、image-aware Prefix Cache |
| 可选能力 | FP8 E4M3 KV；显式 Decode Graph bucket；失败自动回退 Eager |

当前不支持 Qwen3、稠密 Qwen3.5、多图/视频、MTP、量化权重、CPU 权重卸载、TP/EP。项目没有额外自研 CUDA/Triton kernel。

## 最小运行

环境与模型要求以 [快速开始](docs/00_quickstart.md) 和 [pyproject.toml](pyproject.toml) 为准。仓库当前固定使用 PyTorch cu130 运行时。

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
flashinfer download-kernels
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
python example.py
```

Python API：

```python
import os
from nanovllm import LLM, SamplingParams

llm = LLM(
    os.environ["NANOVLLM_MODEL"],
    max_num_batched_tokens=256,
    max_num_seqs=2,
)
try:
    outputs = llm.generate(
        ["请解释稀疏 MoE 的推理过程。"],
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    print(outputs[0]["text"])
finally:
    llm.exit()
```

对话模板、单图 `ImagePrompt`、显存估算、参数含义和 Decode Graph 配置统一见 [快速开始](docs/00_quickstart.md)。

## 验证

单元测试、HF 逐 token 对齐、联合前缀恢复、多模态对齐、benchmark 与 profile 的完整流程统一维护在 [验证与性能分析](docs/03_validation_performance.md)。测试文件索引见 [tests/README.md](tests/README.md)，实验结果记录见 [results/README.md](results/README.md)。

功能存在不等于真实模型已验证通过；性能结果也不能替代数值正确性证据。

## 仓库结构

```text
nanovllm/
  engine/     调度、batch、KV/GDN 资源、缓存与执行
  layers/     Attention、GDN、RoPE、采样
  models/     Qwen3.5-MoE 模型结构与权重映射
  multimodal/ 单图请求与 Processor 展开结果
scripts/      对齐、诊断、benchmark、profile
tests/        单元测试与 GPU 集成测试
docs/         实现、设计和验证文档
results/      可复现实验记录
```

项目沿用 [MIT 许可证](LICENSE)。
