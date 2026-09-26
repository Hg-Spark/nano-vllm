# RTX 5090 目标环境：容量检查与推理基线

[实验导航](README.md) · [指标口径](../docs/03_validation_performance.md)

状态：**待验证（PENDING）**。没有已完成的性能测量。本文件保留原定目标名称，不代表该设备能运行本分支的完整非量化模型。

## 先检查能否装入模型

当前所有文本模型专家权重都加载到单卡，不支持权重量化、CPU 卸载或多卡。按约 35B 参数、BF16 每参数 2 字节估算，权重约为 70 GB（65.2 GiB），还未包括 KV/GDN 状态和临时空间。实际数值以检查点和加载范围为准。

- 实际可用显存：待填写。
- 实际权重规模与 dtype：待填写。
- 完整加载及前向是否成功：待验证。
- 容量不足时的报错与处理：待填写。

若可用显存不足，应记录“环境阻塞”。若改用其他设备或不同权重配置，另建对应基线，不沿用本文件中的设备标签。

## 环境

| 项目 | 记录 |
| --- | --- |
| 日期／执行人 | 待填写 |
| 模型／检查点 revision | 待填写 |
| 实际 GPU 型号／显存 | 待填写；原定目标为 NVIDIA GeForce RTX 5090 |
| Python／PyTorch | 待填写 |
| CUDA／驱动 | 待填写 |
| Triton／Transformers／FlashAttention | 待填写 |
| nano-vLLM 提交／未提交改动 | 待填写 |

## 工作负载

| 项目 | 记录 |
| --- | --- |
| 请求数／输入长度范围／输出长度 | 待填写 |
| `max_num_seqs`／`max_num_batched_tokens` | 待填写 |
| 配置与有效 `max_model_len` | 待填写 |
| KV dtype／K scale／V scale | 待填写；当前 bench CLI 使用默认 auto/1.0/1.0 |
| 前缀缓存是否开启／条目上限 | 待填写 |
| 随机种子／重复次数 | 待填写 |
| 预热和计时范围 | 待填写 |

待执行命令示例（Bash，仓库根目录）：

```bash
python scripts/bench.py "$NANOVLLM_MODEL" \
  --num-requests 8 \
  --min-input-len 64 \
  --max-input-len 256 \
  --output-len 32 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --max-model-len 4096 \
  --seed 0 \
  --json
```

脚本在文本摘要后打印 JSON，不是纯 JSON 输出。输入是随机 token，所有请求同时入队；这不是在线服务流量测试。

## 测量结果

| 指标 | 单位 | 值 |
| --- | --- | --- |
| 请求吞吐 | req/s | 待验证 |
| 输出吞吐 | tok/s | 待验证 |
| 总 token 吞吐 | tok/s | 待验证 |
| prefill 执行吞吐 | tok/s | 待验证 |
| decode 执行吞吐 | tok/s | 待验证 |
| TTFT P50／P99 | ms | 待验证 |
| TPOT P50／P99 | ms | 待验证 |
| E2E P50／P99 | ms | 待验证 |
| 峰值 CUDA 已分配显存 | GiB | 待验证 |

- 原始日志路径：待填写。
- 各次测量结果与波动：待填写。
- 正确性验证记录：待填写。
- 结论及适用范围：待填写。

JSON 的延迟原始单位为秒，填本表时转换为毫秒。少量请求下的 P99 只描述该小样本。
