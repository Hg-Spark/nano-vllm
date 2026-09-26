# Nano-vLLM：Qwen3.5-MoE 混合推理运行时

这个分支专注于 **Qwen3.5-MoE 的单卡、纯文本、eager 推理**，用于学习和验证混合模型的执行过程。核心问题是：同一个请求既有 Full Attention 的 KV 历史，又有 Gated DeltaNet 的循环状态，调度、抢占和前缀复用时，怎样让两者始终对应同一段 token？

```text
Full Attention  → 分页存储的 K/V，随上下文增长
Gated DeltaNet  → 卷积状态 + 循环矩阵，每个活动请求独占状态槽
稀疏 MoE       → 每个 token 动态选择 Top-K 专家，并叠加共享专家
```

**当前 GDN 与 MoE 路径以可读性和数值验证为目标。** 功能已经实现不代表真实模型验证已经通过；[数值对齐](results/parity.md)、[性能基线](results/5090_baseline.md)与[热点分析](results/profile.md)仍是待填写的实验记录。

## 从哪里开始

| 你的目标 | 阅读入口 |
| --- | --- |
| 安装环境、运行模型、理解参数 | [快速开始与配置](docs/00_quickstart.md) |
| 理解 Attention、GDN、MoE 如何协作 | [模型与运行时架构](docs/01_qwen35_hybrid_runtime.md) |
| 理解调度、分块、抢占和前缀缓存 | [调度与混合状态管理](docs/02_scheduler_cache.md) |
| 检查数值正确性、测量性能 | [验证与性能分析](docs/03_validation_performance.md) |
| 解释为什么这样设计 | [与 vLLM、SGLang 的设计取舍](docs/04_vllm_sglang_tradeoffs.md) |
| 对照代码逐步学习、准备项目讲解 | [源码阅读与讲解指南](docs/05_code_walkthrough.md) |

完整阅读顺序见[文档导航](docs/README.md)，后续工作见[路线图](ROADMAP.md)。

## 支持范围

| 方面 | 当前实现 |
| --- | --- |
| 模型 | Qwen3.5-MoE 文本部分，包括 `Qwen3.5-35B-A3B` 对应的模型结构 |
| 权重与执行 | 非量化 safetensors；BF16 为参考验证路径；仅支持单 GPU eager 执行 |
| 模型层 | FlashInfer 分页 Full Attention、局部维度 RoPE、Gated DeltaNet、Top-K 路由与带门控的共享专家 |
| 调度 | 变长打包 prefill、连续批处理、分块 prefill、decode 优先的共享 token 预算 |
| 状态管理 | KV/GDN 联合抢占、执行失败后重算；联合前缀缓存为可选实验能力，默认关闭 |
| 可选缓存 | FP8 E4M3 KV 存储与显式 K/V 缩放系数；FlashInfer 直接读取分页 FP8 KV |
| 生成 | 逐 token 解码、`temperature=0` 的贪心解码、正温度采样、多个 EOS token |

当前不支持 Qwen3、稠密 Qwen3.5、视觉输入、MTP、多模态打包、三轴 mRoPE、量化权重、权重 CPU 卸载、张量/专家并行或 CUDA Graph。融合 GDN、分组/融合 MoE kernel 尚未实现。

## 最小使用示例

在具有足够显存和 CUDA 13.0、Triton、FlashInfer 的环境中，从仓库根目录安装：

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
flashinfer download-kernels
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
python example.py
```

Python 要求为 `>=3.10,<3.13`。模型目录、依赖检查、对话模板和参数说明见[快速开始](docs/00_quickstart.md)。上述命令使用 Bash 语法。

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

这里直接传入原始文本；对话模型的聊天模板需要调用方显式应用，[example.py](example.py) 展示了相应方式。`generate()` 返回生成部分的 `text` 和 `token_ids`，按输入请求顺序排列。

**激活参数量不等于显存需求。** 当前实现将所有专家权重加载到单卡。按约 35B 参数、每参数 2 字节估算，单是权重就约为 70 GB（65.2 GiB），还需要状态和运行空间；这是量级估算，不是实测。FP8 KV 只压缩 KV，不压缩模型权重。文件名中的“5090 基线”也不表示该配置已经能够运行。

## 验证入口

```bash
# 小规模合成配置测试；部分用例需要 CUDA
python -m pytest tests/unit

# 真实权重：先比较 HF 与本实现，再检查 BF16 前缀恢复
python scripts/verify_qwen3_5_moe.py "$NANOVLLM_MODEL" --check-prefix-resume

# 数值不一致时，定位首个超过误差阈值的 decoder 层
python scripts/diagnose_qwen3_5_moe.py "$NANOVLLM_MODEL"
```

单元测试验证局部行为和状态不变量；真实权重上的逐 token 比较检验端到端结果。性能数字不能替代正确性验证。测试前置条件见[测试说明](tests/README.md)。

## 仓库结构

```text
nanovllm/
  engine/     请求、调度、打包、KV/GDN 资源与前缀生命周期
  layers/     Attention、GDN、RoPE、采样与 FP8 KV 存储
  models/     Qwen3.5-MoE 模型结构和权重命名规则
  utils/      执行上下文与权重加载
scripts/      数值对齐、逐层诊断、基准与性能分析
tests/        单元测试和真实权重验证说明
docs/         中文学习文档
results/      实验环境、命令与实测结果
```

项目沿用 [MIT 许可证](LICENSE)。
