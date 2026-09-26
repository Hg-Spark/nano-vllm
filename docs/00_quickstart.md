# 快速开始与配置

[文档导航](README.md) · 下一章：[模型与运行时架构](01_qwen35_hybrid_runtime.md)

本章面向第一次运行这个分支的读者。先检查模型与环境，再运行最小示例，最后调整并发、token 预算和缓存。

## 1. 确认运行条件

依赖声明以 [pyproject.toml](../pyproject.toml) 为准：

| 项目 | 仓库声明或实际要求 |
| --- | --- |
| Python | `>=3.10,<3.13` |
| PyTorch | `>=2.4.0`，必须能够使用 CUDA |
| Triton | `>=3.0.0` |
| Transformers | `>=5.2.0`，须包含所用 Qwen3.5-MoE 配置与参考模型接口 |
| FlashAttention | 仓库未固定版本；需提供代码使用的变长、KV cache 和普通 attention 接口 |
| 权重 | 本地非量化 Qwen3.5-MoE safetensors，参考验证使用 BF16 |
| 设备 | 单 GPU，显存须容纳全部已加载权重、KV、GDN 状态和临时张量 |

最低版本声明不是一份已经实测的兼容版本组合。应在[实验记录](../results/README.md)中记录实际安装的版本。

以下 shell 示例使用 Bash，适用于具备上述依赖的环境。Windows 用户需要自行准备能够运行这些 CUDA 依赖的 Linux/WSL2 环境；本分支没有提供原生 Windows 安装流程。

### 先估算权重显存

`Qwen3.5-35B-A3B` 中的激活参数规模不能用于估算全部权重显存。[模型实现](../nanovllm/models/qwen3_5_moe.py)为所有专家创建参数，[加载器](../nanovllm/utils/loader.py)再加载完整文本模型权重，没有按需卸载专家的实现。

```text
权重字节数 ≈ 实际加载的参数量 × 每个参数的存储字节数
约 35B 参数 × BF16 的 2 字节 ≈ 70 GB ≈ 65.2 GiB
总显存还要加上 KV、GDN 状态、激活和工作空间
```

上述是量级估算，实际值取决于检查点与加载范围。减小并发或启用 FP8 KV 都不能解决“权重本身放不下”的问题。本分支也不支持通过量化权重、CPU 卸载或多卡并行绕过这一限制。

## 2. 获取代码并安装

```bash
git clone --branch feature/qwen35-moe --single-branch https://github.com/Hg-Spark/nano-vllm.git
cd nano-vllm
python -m pip install -e ".[dev]"
```

`-e` 表示使用当前目录中的代码，`[dev]` 额外安装 pytest。CUDA 相关包必须与本机环境兼容。安装完成后检查核心依赖能否导入：

```bash
python -c 'import torch, triton, transformers; from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache; print("CUDA:", torch.cuda.is_available()); print("PyTorch:", torch.__version__); print("Transformers:", transformers.__version__)'
```

`CUDA: False` 或导入失败时，先修复环境；该运行时不会自动切换到 CPU 推理。

## 3. 准备本地模型目录

准备与本分支模型结构匹配的 Qwen3.5-MoE 检查点，例如 `Qwen/Qwen3.5-35B-A3B` 的非量化权重。将模型放在本地目录中，包含：

- `config.json`：模型结构及文本配置；
- 模型所需的 tokenizer 文件；
- 一个或多个 `*.safetensors` 权重分片；
- `generation_config.json`：若存在，会优先从中读取 EOS 配置。

`LLM()` 接收本地目录路径，不会自动下载 Hugging Face 模型 ID。设置示例脚本使用的路径：

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
python example.py
```

[example.py](../example.py)读取该变量并应用聊天模板。`LLM()` 本身不会展开路径中的字面量 `~`，在 Python 中请使用绝对路径或 `os.path.expanduser()`。

## 4. 用中文对话提示运行

```python
import os
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

model_path = os.path.expanduser(os.environ["NANOVLLM_MODEL"])
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请用一个例子解释 KV cache。"}],
    tokenize=False,
    add_generation_prompt=True,
)

llm = LLM(
    model_path,
    max_model_len=4096,
    max_num_seqs=2,
    max_num_batched_tokens=256,
)
try:
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=64),
        use_tqdm=False,
    )
    print(outputs[0]["text"])
    print(outputs[0]["token_ids"])
finally:
    llm.exit()
```

聊天模板负责把角色和内容转成模型期待的文本格式。`generate()` 也接受已经分词的 `list[list[int]]`，这样可以自行控制输入 token。传入多个请求时，可共用一个 `SamplingParams`，也可传入与请求数相等的参数列表。

`exit()` 当前负责同步 CUDA，并让引擎的退出处理只执行一次；它不承诺立即释放所有仍被 Python 对象引用的模型张量。

## 5. 参数怎样影响执行

以下默认值对应 [Config](../nanovllm/config.py)。

| 参数 | 默认值 | 含义与影响 |
| --- | --- | --- |
| `max_num_batched_tokens` | `512` | 一轮调度共享的 token 预算；先满足 decode，余量给 prefill |
| `max_num_seqs` | `4` | 一轮请求数上限，也是活动 GDN 状态槽数量；分块 prefill 也占槽 |
| `max_model_len` | `4096` | 单请求输入与最大输出的总 token 上限，还会被模型位置上限截断 |
| `gpu_memory_utilization` | `0.9` | KV 容量估算使用的显存预算比例，范围 `(0,1]`；不是权重加载的硬上限 |
| `kvcache_block_size` | `256` | 每个 KV 块的 token 容量；使用正的 256 倍数 |
| `max_prefix_cache_entries` | `16` | 联合前缀条目数上限；`0` 关闭前缀缓存，不关闭请求自身的 KV/GDN 状态 |
| `kv_cache_dtype` | `"auto"` | `auto` 跟随模型 dtype；`fp8_e4m3` 使用 FP8 KV 存储 |
| `kv_cache_k_scale` | `1.0` | FP8 K 的全局标量缩放系数，必须有限且大于零 |
| `kv_cache_v_scale` | `1.0` | FP8 V 的全局标量缩放系数，必须有限且大于零 |
| `tensor_parallel_size` | `1` | 当前仅支持 `1` |
| `enforce_eager` | `True` | 当前必须为 `True` |

物理 KV 块数由启动时的显存预算推导，不是构造参数。未知参数会触发 `unsupported runtime options`，不能直接照搬其他推理引擎的配置。

[SamplingParams](../nanovllm/sampling_params.py)只有以下三项：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `temperature` | `1.0` | `0` 使用贪心选择；正值进行温度采样，数值对齐建议用 `0` |
| `max_tokens` | `64` | 最多生成的 token 数，必须大于零 |
| `ignore_eos` | `False` | 为 `True` 时忽略 EOS，以固定长度生成；主要用于验证和测量 |

当前没有 `top_p` 或采样 `top_k` 参数。结束符优先读取 `generation_config.json`，再回退到 tokenizer 和文本配置，可接受多个 EOS token ID。

### 一个调参例子

假设 `max_num_batched_tokens=8`，本轮选中了 3 个 decode 请求，它们各消费 1 个 token，prefill 最多还可获得 5 个 token。还需要满足请求数、GDN 槽位和 KV 容量约束。

增加预算可能让长输入更快完成，也可能增加其他请求等待一次 forward 的时间。增加 `max_num_seqs` 会预留更多 GDN 状态，留给 KV 的预算可能减少。应通过[基准测量](03_validation_performance.md)判断取舍。

## 6. 常见问题

| 现象 | 含义与处理入口 |
| --- | --- |
| `model path does not exist` | 检查本地目录与路径展开；不能直接传模型仓库 ID |
| `only Qwen3.5-MoE checkpoints are supported` | 检查 `model_type`，不要混用稠密 Qwen3.5 或 Qwen3 |
| 权重加载时 OOM | 首先核对完整权重显存；降低 KV 预算不能缩小权重 |
| `insufficient GPU memory ... recurrent state pools` | 权重与状态预留后没有足够 KV 容量；检查并发、显存占用及模型规模 |
| `prompt + completion exceeds max_model_len` | 按 token 数检查 `输入长度 + max_tokens`，调整长度或有效上下文上限 |
| `scheduler could not make progress` | 当前轮次无法安排工作；检查 KV 容量、状态槽及请求长度，不能假设分块就能无限扩展上下文 |
| 未出现前缀命中 | 检查精确 token 前缀、块边界、条目是否被淘汰，以及是否至少留一个 token 重算，见[第 2 章](02_scheduler_cache.md) |
| FP8 未校准告警 | 两个 scale 都为 `1.0` 时会提示；需要针对检查点验证精度，见[第 3 章](03_validation_performance.md) |
| 文本看起来正常但验证失败 | 比较 token ID 和逐层误差；“读起来合理”不足以证明实现正确 |

成功生成之后，请继续运行[真实权重验证](../tests/integration/README.md)，并将环境与结果写入 `results/`。
