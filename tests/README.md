# 测试说明

[项目首页](../README.md) · [完整验证流程](../docs/03_validation_performance.md)

这里维护测试入口、运行条件和测试文件职责。真实权重如何验证、如何记录性能，统一见 [验证与性能分析](../docs/03_validation_performance.md)。

## 运行

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit
```

单元测试不下载真实权重，但测试收集会导入项目运行时，因此仍可能依赖 torch、Transformers、Triton 和 FlashInfer。CUDA/BF16 专用用例带能力检查；报告结果时应分别记录通过、失败和跳过数量。

## 单元测试索引

| 文件 | 主要检查 |
| --- | --- |
| [test_config.py](unit/test_config.py) | Qwen3.5-MoE admission、mRoPE 配置、Graph bucket 与 KV page 参数 |
| [test_qwen3_5_moe.py](unit/test_qwen3_5_moe.py) | MoE 路由/专家计算、权重变换、模型 cache topology |
| [test_gated_delta_net.py](unit/test_gated_delta_net.py) | GDN state layout、Chunk/Decode 连续性、变长状态对齐、snapshot 恢复 |
| [test_attention.py](unit/test_attention.py) | FlashInfer wrapper 调用契约、Paged KV 与 FP8 scale |
| [test_context.py](unit/test_context.py) | Batch execution Context 的安装、读取与恢复 |
| [test_prefill_batch.py](unit/test_prefill_batch.py) | Packed Prefill、mRoPE、Paged KV 与 state slot 对齐 |
| [test_multimodal.py](unit/test_multimodal.py) | 单图展开、视觉 span、mRoPE 与 image-aware Prefix |
| [test_state_manager.py](unit/test_state_manager.py) | GDN slot 分配、所有权与释放 |
| [test_hybrid_resources.py](unit/test_hybrid_resources.py) | KV/GDN 联合资源不变量与 accounting |
| [test_scheduler.py](unit/test_scheduler.py) | Decode-first budget、Chunked Prefill、Continuous Batching |
| [test_scheduler_recovery.py](unit/test_scheduler_recovery.py) | 抢占、资源不足和 deadlock recovery |
| [test_scheduler_prefix_cache.py](unit/test_scheduler_prefix_cache.py) | Prefix restore、snapshot 边界与 postprocess 原子性 |
| [test_prefix_cache.py](unit/test_prefix_cache.py) | Prefix 最长匹配、LRU 与 KV 引用 |
| [test_runtime_warmup.py](unit/test_runtime_warmup.py) | Decode/Vision/Prefill warmup 与 token budget |
| [test_llm_engine.py](unit/test_llm_engine.py) | 请求校验、step 提交、异常恢复、Decode Graph fallback |
| [test_fp8_kv_cache.py](unit/test_fp8_kv_cache.py) | FP8 KV dtype、scale 与写入 |
| [test_rotary_embedding.py](unit/test_rotary_embedding.py) | RoPE 位置变换 |
| [test_sampler.py](unit/test_sampler.py) | 贪心与温度采样 |

真实分页 Attention 的 GPU 数值测试位于 [integration/test_flashinfer_attention.py](integration/test_flashinfer_attention.py)。

真实检查点验证通过 `scripts/verify_*.py` 执行；入口说明见 [integration/README.md](integration/README.md)。单元测试通过不代表真实检查点已经对齐。
