# 安装与运行

本页维护安装、模型准备、交互与配置，测试与性能命令见[验证手册](validation.md)。

## 确认运行条件

依赖声明以 [pyproject.toml](../pyproject.toml) 为准：

| 项目 | 参考版本、依赖声明或运行要求 |
| --- | --- |
| Python | 推荐 `3.12`；项目不增加运行时版本检查 |
| PyTorch | 参考 `2.14.0`；依赖不固定版本 |
| CUDA | 参考 `13.0`（PyTorch `cu130` 构建）；启动时不检查版本 |
| Transformers | `>=5.19.0`，须包含所用 Qwen3.5 稠密/MoE 配置与参考模型接口 |
| FlashInfer | `flashinfer-python==0.7.0`；Full Attention 统一使用分页 prefill/decode wrapper |
| 权重 | 本地非量化 Qwen3.5 稠密/MoE 完整 safetensors（文本 + visual），稠密 2B 使用 FP16 验证 |
| 设备 | 单 GPU，显存须容纳全部已加载权重、KV、GDN 状态和临时张量 |

统一使用 Python 3.12 作为开发、CI 与基准测试参考环境，不额外添加运行时 Python 版本检查。PyTorch 和 CUDA 的参考版本仅供安装和复现参考，不表示这组环境已经完成全部真实 GPU 验证。所选环境仍需满足依赖包自身的要求；应在本地实验输出中记录实际安装的版本。

以下 shell 示例使用 Bash，适用于具备上述依赖的环境。Windows 用户需要自行准备能够运行这些 CUDA 依赖的 Linux/WSL2 环境；本分支没有提供原生 Windows 安装流程。

### 先估算权重显存

`Qwen3.5-35B-A3B` 中的激活参数规模不能用于估算全部权重显存。[模型实现](../nanovllm/models/qwen3_5.py)为所有专家和视觉塔创建参数，[加载器](../nanovllm/utils/loader.py)严格加载完整检查点，没有按需卸载专家或视觉权重的实现。

```text
权重字节数 ≈ 实际加载的参数量 × 每个参数的存储字节数
约 35B 参数 × BF16 的 2 字节 ≈ 70 GB ≈ 65.2 GiB
总显存还要加上 KV、GDN 状态、激活和工作空间
```

上述是量级估算，实际值取决于检查点与加载范围。减小并发或启用 FP8 KV 都不能解决“权重本身放不下”的问题。本分支也不支持通过量化权重、CPU 卸载或多卡并行绕过这一限制。

## 获取代码并安装

```bash
git clone --branch qwen3.5 --single-branch https://github.com/Hg-Spark/nano-vllm.git
cd nano-vllm

python -m pip install -e ".[dev]"
flashinfer download-kernels
```

`-e` 表示使用当前目录中的代码，`[dev]` 额外安装 pytest。无 GPU 环境也可先安装项目和依赖；模型推理与 GPU 性能测试需要 CUDA GPU。如需记录安装版本，可执行：

```bash
python -c 'import torch, flashinfer, transformers; print("PyTorch:", torch.__version__); print("PyTorch CUDA:", torch.version.cuda); print("FlashInfer:", flashinfer.__version__); print("Transformers:", transformers.__version__)'
```

运行时固定 FlashInfer 0.7.0 GDN Packed Prefill 和 State-Indexed Decode，复用 FP32 `[V,K]` State Pool。初始化时要求 SM90/SM100/SM120 且 K/V head_dim 均为 128，否则直接报错；RTX 2080 Ti / SM75 不再支持。仍须完成真实权重对齐与 GPU Profile。

## 准备本地模型目录

准备与本分支模型结构匹配的 Qwen3.5 检查点，例如稠密 `Qwen/Qwen3.5-2B` 或 MoE `Qwen/Qwen3.5-35B-A3B` 的非量化权重。将模型放在本地目录中，包含：

- `config.json`：模型结构及文本配置；
- 模型所需的 tokenizer 文件；
- 一个或多个 `*.safetensors` 权重分片；
- `generation_config.json`：若存在，会优先从中读取 EOS 配置。

`LLM()` 接收本地目录路径，不会自动下载 Hugging Face 模型 ID。设置示例脚本使用的路径：

```bash
export NANOVLLM_MODEL=/absolute/path/to/Qwen3.5-MoE
export NANOVLLM_DTYPE=bfloat16
python example.py
```

[example.py](../example.py)读取两个环境变量 `NANOVLLM_MODEL`、`NANOVLLM_DTYPE` 并应用聊天模板。`LLM()` 本身不会展开路径中的字面量 `~`，在 Python 中请使用绝对路径或 `os.path.expanduser()`。

## 用中文对话提示运行

复用本地稠密 2B 的 WSL 示例（FP16、Eager）：

```bash
python -m scripts.chat "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --max-model-len 512 --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.9 --max-new-tokens 64 --temperature 0
```

稠密模型使用 `intermediate_size` 构造普通 SwiGLU FFN，不需要专家参数或后端选择。其 `tie_word_embeddings` 会将 embedding 和输出层绑定为同一参数，加载器接受检查点只保存其中一份权重。

交互式多轮对话：

```bash
python -m scripts.chat "$NANOVLLM_MODEL" --dtype "$NANOVLLM_DTYPE" \
  --system "你是一个简洁、准确的中文助手。" \
  --max-new-tokens 256 --temperature 0.7
```

省略模型参数时读取 `NANOVLLM_MODEL`，统一使用模块入口 `python -m scripts.chat`。
脚本加载一次模型，使用模型 tokenizer 的聊天模板编码完整对话历史，每轮生成结束后显示回复。
支持 `/clear` 清空对话并保留 system 提示、`/help` 查看命令、`/exit` 或 `/quit` 退出；Ctrl+C 和 EOF 也会结束会话并关闭引擎。
上下文长度或物理 KV 容量不足时会拒绝当前输入并提示清空历史，已完成的对话不受影响。
可用 `--max-model-len`、`--max-num-batched-tokens`、`--gpu-memory-utilization` 调整资源。MoE 统一使用 FlashInfer，受支持的 GPU 可通过 `--decode-graph` 请求 B=1 Decode Graph。

通过 Python API 构造单轮提示：

```python
import os
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

model_path = os.path.expanduser(os.environ["NANOVLLM_MODEL"])
tokenizer = AutoTokenizer.from_pretrained(model_path)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请用一个例子解释 KV cache。"}],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
)

llm = LLM(
    model_path,
    dtype=os.environ["NANOVLLM_DTYPE"],
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

聊天模板直接生成 Token ID，避免字符串输入路径再次注入特殊 Token。`generate()` 也接受已经分词的 `list[list[int]]`，这样可以自行控制输入 token。传入多个请求时，可共用一个 `SamplingParams`，也可传入与请求数相等的参数列表。

单图输入使用 `ImagePrompt(text, image)`。先用 `AutoProcessor.apply_chat_template(..., tokenize=False)` 生成含一个图像 placeholder 的文本，再把同一张 PIL 图片交给 `ImagePrompt`；Runtime 会用 Processor 一次性展开 token、网格、三轴位置并在展开后执行长度/KV 容量检查。多图和视频会被拒绝。

`exit()` 当前负责同步 CUDA，并让引擎的退出处理只执行一次；它不承诺立即释放所有仍被 Python 对象引用的模型张量。

模型的 `dtype` 是**必填参数**，仅允许 `"float16"`、`"bfloat16"`。不从检查点推断，不支持模型 dtype 为 `"auto"`。这与 KV 存储参数 `kv_cache_dtype="auto"` 不同：后者表示 KV 采用当前模型计算精度。

## 参数怎样影响执行

以下默认值对应 [Config](../nanovllm/config.py)。

| 参数 | 默认值 | 含义与影响 |
| --- | --- | --- |
| `max_num_batched_tokens` | `512` | 一轮调度共享的 token 预算；先满足 decode，余量给 prefill |
| `max_num_seqs` | `4` | 一轮请求数上限，也是活动 GDN 状态槽数量；分块 prefill 也占槽 |
| `max_model_len` | `4096` | 单请求输入与最大输出的总 token 上限，还会被模型位置上限截断 |
| `gpu_memory_utilization` | `0.8` | KV 容量估算使用的显存预算比例，范围 `(0,1]`；不是权重加载的硬上限 |
| `kv_cache_block_size` | `16` | FlashInfer page size；当前允许 `16/32/64/128`，基线使用 16 |
| `max_prefix_cache_entries` | `0` | 联合前缀缓存默认关闭；设为正数后限制缓存条目数，不影响请求自身的 KV/GDN 状态 |
| `max_prefix_cache_snapshot_bytes` | `None` | CPU GDN 快照字节上限；`0` 禁止发布快照 |
| `max_visual_tokens_per_request` | `0` | 显式限制单请求图像视觉 Token 数；准入时检查，并按上限预留特征缓存。`None` 使用 `max_model_len`，`0` 禁用图像请求 |
| `kv_cache_dtype` | `"auto"` | `auto` 跟随模型 dtype；`fp8_e4m3` 使用 FP8 KV 存储 |
| `kv_cache_k_scale` | `1.0` | FP8 K 的全局标量缩放系数，必须有限且大于零 |
| `kv_cache_v_scale` | `1.0` | FP8 V 的全局标量缩放系数，必须有限且大于零 |
| `decode_graph_batch_sizes` | `()` | 显式启用的 Decode CUDA Graph batch size；全部 bucket 与 Eager wrapper 共用 FlashInfer workspace，未配置的 batch size 使用 Eager；捕获时记录显存增量和余量 |

MoE Prefill 与 Decode 均使用 FlashInfer CUTLASS Fused MoE，并共享按
`max(max_num_batched_tokens, max_num_seqs)` 分配的工作空间。受支持的 GPU
架构之外直接报错；稠密 Qwen3.5 不创建 MoE 工作空间。GDN 对 SM90/SM100/SM120 以外的 GPU 同样不提供计算回退。
数值对照使用独立的 Hugging Face 验证脚本，不保留内部 MoE 参考后端。

运行时固定为单卡。Prefill 始终使用 Eager；Decode Graph 默认关闭，只捕获模型前向与 logits，Sampler、调度提交及 Prefix Snapshot 仍在 Graph 外；执行失败时引擎 Fail-Fast。

物理 KV 块数由启动时的显存预算推导，不是构造参数。未知参数会触发 dataclass 构造时的 `TypeError`，不能直接照搬其他推理引擎的配置。

### 显存预算与启动预执行

当前采用**静态定容 + 运行时逻辑 Block 按需分配**。模型权重和 MoE/FlashInfer Workspace 就位后，按剩余显存扣除 GDN 与视觉特征预留，详细公式和 Block 布局见[运行时缓存分配](runtime.md#启动时怎样分配缓存)。

默认比例为 `0.8`，不再运行专用的 Attention SDPA / GDN 临时状态 Forward 去测量激活峰值。余下显存供激活、CUDA Graph 等使用，**不是严格的最坏情况保证**。如果 Prefill OOM，优先降低 `max_num_batched_tokens` 或 `gpu_memory_utilization`；Decode OOM 可进一步降低 `max_num_seqs`。需要做最大负载测试以决定目标 GPU 上的合理配置，步骤见 [验证手册](validation.md#运行时显存预算验证无需新增后端)。

`decode_graph_batch_sizes` 非空时，Graph Capture 仍会先用固定输入运行一次模型以触发必要的 Kernel 初始化/编译，清理 KV/GDN 状态后再捕获；它与已经移除的**启动显存 Profiling Warmup**不是同一机制。未配置的 batch size 使用 Eager；已配置的 bucket 捕获或运行失败会直接抛异常，避免静默回退。

[SamplingParams](../nanovllm/sampling_params.py)只有以下三项：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `temperature` | `1.0` | `0` 使用贪心选择；正值进行温度采样，数值对齐建议用 `0` |
| `max_tokens` | `64` | 最多生成的 token 数，必须大于零 |
| `ignore_eos` | `False` | 为 `True` 时忽略 EOS，以固定长度生成；主要用于验证和测量 |

当前没有 `top_p` 或采样 `top_k` 参数。结束符优先读取 `generation_config.json`，再回退到 tokenizer 和文本配置，可接受多个 EOS token ID。

### 一个调参例子

假设 `max_num_batched_tokens=8`，本轮选中了 3 个 decode 请求，它们各消费 1 个 token，prefill 最多还可获得 5 个 token。还需要满足请求数、GDN 槽位和 KV 容量约束。

增加预算可能让长输入更快完成，也可能增加其他请求等待一次 forward 的时间。增加 `max_num_seqs` 会预留更多 GDN 状态，留给 KV 的预算可能减少。应通过[基准测量](validation.md)判断取舍。

## 常见问题

| 现象 | 含义与处理入口 |
| --- | --- |
| `model path does not exist` | 检查本地目录与路径展开；不能直接传模型仓库 ID |
| `only Qwen3.5 dense or MoE checkpoints are supported` | 检查 Qwen3.5 的 `model_type`；Qwen3 不支持 |
| 权重加载时 OOM | 首先核对完整权重显存；降低 KV 预算不能缩小权重 |
| 首次 Prefill / Decode 时 OOM | 静态定容不会测量最坏激活峰值；降低 `gpu_memory_utilization`、Token Budget 或并发量，并按边界负载重新测量 |
| `insufficient GPU memory ... recurrent state pools` | 权重与状态预留后没有足够 KV 容量；检查并发、显存占用及模型规模 |
| `requested prompt (...) + max_tokens (...) exceeds max_model_len (...)` | 按 token 数检查 `输入长度 + max_tokens`，调整长度或有效上下文上限 |
| `scheduler could not make progress` | 当前轮次无法安排工作；检查 KV 容量、状态槽及请求长度，不能假设分块就能无限扩展上下文 |
| 未出现前缀命中 | 检查精确 token 前缀、完整块边界、条目是否被淘汰，以及缓存边界后是否仍有至少一个 token 待计算，见[第 2 章](scheduling.md) |
| FP8 未校准告警 | 两个 scale 都为 `1.0` 时会提示；需要针对检查点验证精度，见[第 3 章](validation.md) |
| 图像 token 数/特征数不一致 | 检查是否使用 Qwen3.5 官方 Processor/chat template，以及展开前是否恰有一个 image placeholder |
| Decode Graph 配置执行失败 | 配置的 bucket 捕获、Plan 或布局不匹配会抛异常；检查显存、FlashInfer 支持及 GPU 对齐结果 |
| 文本看起来正常但验证失败 | 比较 token ID 和逐层误差；“读起来合理”不足以证明实现正确 |

成功生成之后，请继续运行[验证手册](validation.md)，并将环境与结果保存在本地未跟踪的 `results/`。
