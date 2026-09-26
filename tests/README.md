# 测试说明

[项目首页](../README.md) · [验证与性能分析](../docs/03_validation_performance.md)

测试分成小规模合成配置的单元测试，以及需要真实检查点的集成验证。两者承担不同的证明任务。

## 运行条件

先安装项目与开发依赖：

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit
```

单元测试不下载真实权重，但会导入项目运行时。即使某条测试只操作 CPU 数据，收集测试时也可能需要 torch、Transformers、Triton 和 FlashAttention。不能把整个测试集描述为无需 CUDA 软件依赖的纯 CPU 套件。

[test_fp8_kv_cache.py](unit/test_fp8_kv_cache.py)中的 GPU 读写测试带 CUDA 可用性检查，无可用 CUDA 时会跳过。报告结果时分别列出通过、失败和跳过数量；导入失败不是测试通过。

## 单元测试覆盖

| 文件 | 主要检查 |
| --- | --- |
| [test_qwen3_5_moe.py](unit/test_qwen3_5_moe.py) | 模型配置、MoE 路由与专家计算、权重映射 |
| [test_gated_delta_net.py](unit/test_gated_delta_net.py) | 分块/逐 token 连续性、变长请求状态对齐、BF16 快照 |
| [test_rotary_embedding.py](unit/test_rotary_embedding.py) | RoPE 的维度与位置变换 |
| [test_sampler.py](unit/test_sampler.py) | 贪心与采样分支 |
| [test_prefill_batch.py](unit/test_prefill_batch.py) | query/key 长度、位置、槽位与打包边界 |
| [test_state_manager.py](unit/test_state_manager.py) | 槽位分配、所有权验证与释放 |
| [test_scheduler.py](unit/test_scheduler.py) | 预算、decode 轮转、增量 KV、分块与抢占 |
| [test_prefix_cache.py](unit/test_prefix_cache.py) | 精确前缀匹配、LRU、KV 引用与快照边界 |
| [test_llm_engine.py](unit/test_llm_engine.py) | 请求校验、EOS、批次执行及异常恢复 |
| [test_fp8_kv_cache.py](unit/test_fp8_kv_cache.py) | dtype、显式 scale、分页物化、多块 GQA 读取 |

局部运行示例：

```bash
python -m pytest tests/unit/test_scheduler.py tests/unit/test_prefix_cache.py
```

## 真实权重验证

```bash
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" --check-prefix-resume
```

详细条件和流程见[集成验证说明](integration/README.md)。单元测试通过不等于真实检查点通过；基准吞吐提高也不构成数值正确性证据。
