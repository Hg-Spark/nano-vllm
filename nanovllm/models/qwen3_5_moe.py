"""Qwen3.5 sparse-MoE router, packed experts and shared expert block."""

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.models.qwen3_5_layers import Qwen3_5MLP


class Qwen3_5MoeExperts(nn.Module):
    """FlashInfer routed experts with checkpoint [gate, up] reordered to [up, gate]."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_size,
                self.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
            )
        )
        self._moe_workspace: torch.Tensor | None = None
        self._moe_output: torch.Tensor | None = None

    def bind_runtime_buffers(
        self,
        workspace: torch.Tensor | None,
        output: torch.Tensor | None,
    ) -> None:
        self._moe_workspace = workspace
        self._moe_output = output

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._moe_workspace is None or self._moe_output is None:
            raise RuntimeError("FlashInfer MoE runtime buffers are not initialized")
        num_tokens = hidden_states.size(0)
        if num_tokens > self._moe_output.size(0):
            raise RuntimeError("MoE token batch exceeds allocated workspace capacity")
        from flashinfer.fused_moe import cutlass_fused_moe

        result = cutlass_fused_moe(
            hidden_states,
            selected_experts.to(torch.int32),
            routing_weights.float(),
            self.gate_up_proj,
            self.down_proj,
            hidden_states.dtype,
            quant_scales=[],
            output=self._moe_output[:num_tokens],
            use_fused_finalize=False,
            tune_max_num_tokens=self._moe_output.size(0),
            enable_pdl=False,
            workspace_buffer=self._moe_workspace,
        )
        return result[0] if isinstance(result, (list, tuple)) else result


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.hidden_size,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(
            router_logits,
            dtype=torch.float32,
            dim=-1,
        )
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        )
        return (
            routing_weights.to(router_logits.dtype),
            selected_experts,
        )


class Qwen3_5SparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MLP(
            config,
            config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        shared_output = self.shared_expert(hidden_states)
        shared_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output

        routing_weights, selected_experts = self.gate(hidden_states)
        routed_output = self.experts(
            hidden_states,
            selected_experts,
            routing_weights,
        )
        return (routed_output + shared_output).reshape(original_shape)


def initialize_moe_runtime(model, config) -> None:
    """Allocate shared prefill/decode FlashInfer MoE buffers."""
    model_dtype = next(model.parameters()).dtype
    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if arch not in {89, 90, 100, 103, 107, 110, 120, 121}:
        raise RuntimeError(f"FlashInfer MoE is unsupported on SM{arch}")
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

    cfg = config.text_config
    max_tokens = max(config.max_num_batched_tokens, config.max_num_seqs)
    try:
        size = cutlass_fused_moe_workspace_size(
            max_tokens,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            cfg.num_experts_per_tok,
            x_dtype=model_dtype,
            weight_dtype=model_dtype,
            output_dtype=model_dtype,
            use_fused_finalize=False,
            device=torch.device("cuda"),
        )
        workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
        output = torch.empty(max_tokens, cfg.hidden_size, dtype=model_dtype, device="cuda")
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("FlashInfer MoE workspace allocation ran out of memory") from exc
    except (NotImplementedError, RuntimeError) as exc:
        raise RuntimeError(f"FlashInfer MoE is unavailable on this runtime: {exc}") from exc
    model.bind_moe_runtime(workspace, output)
