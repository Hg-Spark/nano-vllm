# nano-vLLM — Qwen3.5 Hybrid Runtime

面向 **Qwen3.5 Dense / MoE** 的单 GPU 推理实验实现。参考环境为 Python 3.12；运行时同时管理 Full Attention 的 Paged KV 与 Gated DeltaNet 的 Conv / Recurrent State，支持 Dense SwiGLU 或 FlashInfer CUTLASS Fused MoE。

## 支持范围

| 模块 | 当前实现 |
| --- | --- |
| 模型 | Qwen3.5 Dense / MoE、非量化完整 safetensors、单 GPU；不支持 Qwen3、TP/EP、CPU 卸载、量化权重 |
| 计算 | FlashInfer Paged Attention / KV Append；FlashInfer Packed GDN Prefill / Indexed Decode（SM90/100/120）；打包 Dense FFN / FlashInfer CUTLASS MoE；Gemma Fused Add RMSNorm、mRoPE 与 Sampling |
| 调度 | Continuous Batching、Decode-first token budget、State-aware Chunked Prefill |
| 状态 | Paged KV、GDN State Slot、全生命周期额度准入、可选联合 Prefix Cache |
| 输入 | 文本与单图（Processor / 视觉塔），不支持多图/视频 |
| 可选 | FP8 E4M3 KV、固定 Decode CUDA Graph bucket；Prefill Eager |

GDN 循环矩阵及快照保持 FP32 `[V,K]` 布局；Prefill/Decode 统一使用 FlashInfer 0.7.0，要求 SM90/100/120 且 K/V head_dim=128。无 PyTorch GDN 计算回退，SM75（例如 RTX 2080 Ti）不支持推理。输入投影和 Dense MLP 的 Gate/Up 权重在 checkpoint 加载后打包。MoE Prefill / Decode 统一使用 FlashInfer CUTLASS。模型 dtype 必须显式为 `float16` 或 `bfloat16`。

KV 写入统一调用 FlashInfer `append_paged_kv_cache`；FP8 路径先用 PyTorch 进行 FP32 缩放、裁剪及转换，再追加到分页缓存。项目不再直接导入 Triton 或维护 Triton kernel；依赖包可能仍间接安装或使用 Triton。

**执行边界：** Prefill 使用 FlashInfer Paged Attention Eager；Decode 根据 Batch size 选择固定 CUDA Graph 或 Eager。启动时以 `gpu_memory_utilization=0.8`（默认）进行静态 KV 定容，**不执行无缓存 Attention/GDN Profiling Warmup**；启用 Decode Graph 时仍需预执行、捕获并清理临时写入的状态。物理 KV Pool 启动时预分配，逻辑 Block 在请求运行期间按需分配。显存余量不是任意负载均不会 OOM 的保证。默认 `max_visual_tokens_per_request=0`，仅加载文本模型；图文推理需显式设置正数视觉 Token 上限，或设为 `None` 使用保守上限。启用 Graph 后记录每个 Bucket 的显存变化。

## 最小运行

```bash
git clone --branch qwen3.5 --single-branch https://github.com/Hg-Spark/nano-vllm.git
cd nano-vllm
python -m pip install -e ".[dev]"
flashinfer download-kernels
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5
export NANOVLLM_DTYPE=float16
python example.py
python -m scripts.chat "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE"
```

依赖以 [pyproject.toml](pyproject.toml) 为准；FlashInfer 固定 0.7.0。完整说明见 [安装与运行](docs/getting-started.md)。

## 文档

- [模型与运行时](docs/runtime.md)：模型数学、批次布局、Forward 与代码入口。
- [调度与缓存](docs/scheduling.md)：KV/GDN 状态不变量、额度预留、提交、恢复与前缀缓存。
- [验证与性能](docs/validation.md)：测试、数值对齐、Benchmark、Profile 和证据边界。
- [设计取舍](docs/design-decisions.md)：vLLM / SGLang 参考与简化的理由。
- [路线图](ROADMAP.md) 与 [验证方法](docs/validation.md)。

## 提交与源码导读

相对 `main` 的重构历史按职责整理为主题提交（后续修改也继续按主题提交）：`feat(model/cache/runtime)` 构成模型与执行主链；`feat(cli/validation)` 提供实验入口；`test(model/kernels/scheduler/engine/cli)` 验证模块契约；`docs` 收录设计与证据。可在 [qwen3.5 提交历史](https://github.com/Hg-Spark/nano-vllm/commits/qwen3.5/)查看最新顺序；若本地也已获取 `main` 引用，可运行 `git log --oneline main..qwen3.5`。旧版 Warmup、参考后端与重复实现均未作为独立路径保留。

建议依次阅读 [模型与运行时](docs/runtime.md) → [调度与缓存](docs/scheduling.md) → [设计取舍](docs/design-decisions.md) → [验证与性能](docs/validation.md)，避免把未运行的 GPU 验证当作功能已经通过实测。

**验证边界：** 仓库不再跟踪历史实测结果。更改 FlashInfer RoPE、Gemma Fused Add RMSNorm 与 Sampling 后，应按[验证手册](docs/validation.md)重新验证 HF Greedy、Chunked Prefill、图文 mRoPE 和 Decode Graph；未提供的新 GPU 结果不视为已通过。

本项目沿用 [MIT 许可证](LICENSE)。
