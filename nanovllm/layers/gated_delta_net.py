import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from nanovllm.layers.gdn_chunk import chunk_gated_delta_rule
from nanovllm.layers.gdn_state import GDNStatePool
from nanovllm.utils.context import get_context


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        x = self.weight * x.to(input_dtype) * F.silu(gate.float())
        return x.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Correctness-first Qwen3.5 Gated DeltaNet implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self._flashinfer_device_major = None
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.state_pool = GDNStatePool(
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )

    def pack_projections(self) -> None:
        # Checkpoint weights are loaded with their original HF names first.
        # Pack once so prefill and decode share a single input GEMM.
        parts = (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a)
        packed = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.value_dim + 2 * self.num_v_heads,
            bias=False,
            device=parts[0].weight.device,
            dtype=parts[0].weight.dtype,
        )
        with torch.no_grad():
            offset = 0
            for part in parts:
                end = offset + part.weight.size(0)
                packed.weight[offset:end].copy_(part.weight)
                offset = end
        self.in_proj = packed
        del self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a

    def _causal_conv(self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        seq_len = mixed_qkv.size(0)
        x = mixed_qkv.transpose(0, 1).unsqueeze(0)
        x_with_state = torch.cat((conv_state.unsqueeze(0), x), dim=-1).to(self.conv1d.weight.dtype)
        conv_state.copy_(x_with_state[0, :, -self.conv_kernel_size :])
        out = F.conv1d(
            x_with_state, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim
        )
        return F.silu(out[:, :, -seq_len:]).squeeze(0).transpose(0, 1).to(mixed_qkv.dtype)

    def _delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_v_heads != self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=1)
            key = key.repeat_interleave(repeats, dim=1)
        output, final_state = chunk_gated_delta_rule(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            decay.unsqueeze(0),
            beta.unsqueeze(0),
            recurrent_state.transpose(-1, -2).unsqueeze(0).contiguous(),
        )
        recurrent_state.copy_(final_state[0].transpose(-1, -2))
        return output[0]

    def _project_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_qkv, z, b, a = self.in_proj(hidden_states).split(
            (self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads),
            dim=-1,
        )
        z = z.view(-1, self.num_v_heads, self.head_v_dim)
        return mixed_qkv, z, b, a

    def _gates(self, b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = b.sigmoid().float()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        return beta, decay

    def _can_use_flashinfer(self, hidden_states: torch.Tensor) -> bool:
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or self.head_k_dim != 128
            or self.head_v_dim != 128
        ):
            return False
        if self._flashinfer_device_major is None:
            self._flashinfer_device_major = torch.cuda.get_device_capability(hidden_states.device)[
                0
            ]
        return self._flashinfer_device_major in (9, 10, 12)

    def _split_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        return (
            query.view(-1, self.num_k_heads, self.head_k_dim),
            key.view(-1, self.num_k_heads, self.head_k_dim),
            value.view(-1, self.num_v_heads, self.head_v_dim),
        )

    def _finalize_output(self, core: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.view(-1, self.value_dim))

    def _forward_prefill_batch(
        self,
        hidden_states: torch.Tensor,
        state_slots: tuple[int, ...],
        state_prefix_lens: tuple[int, ...],
        q_offsets: tuple[int, ...],
    ) -> torch.Tensor:
        mixed_qkv, z, b, a = self._project_inputs(hidden_states)
        beta, decay = self._gates(b, a)
        use_flashinfer = self._can_use_flashinfer(hidden_states)
        core_parts = []
        conv_parts = []
        for seq_idx, slot_id in enumerate(state_slots):
            q_start, q_end = q_offsets[seq_idx], q_offsets[seq_idx + 1]
            expected_prefix = state_prefix_lens[seq_idx]
            if q_end <= q_start:
                raise RuntimeError(f"invalid GDN prefill range for sequence {seq_idx}")
            if expected_prefix < 0:
                raise RuntimeError(f"invalid GDN state prefix for sequence {seq_idx}")
            if expected_prefix == 0:
                self.state_pool.clear(slot_id)
            conv_state, recurrent_state = self.state_pool.slot(slot_id)
            convolved = self._causal_conv(mixed_qkv[q_start:q_end], conv_state)
            if use_flashinfer:
                conv_parts.append(convolved)
            else:
                query, key, value = self._split_qkv(convolved)
                with record_function("nanovllm::gdn_prefill_hf_chunk"):
                    core_parts.append(
                        self._delta_rule(
                            query,
                            key,
                            value,
                            decay[q_start:q_end],
                            beta[q_start:q_end],
                            recurrent_state,
                        )
                    )

        if use_flashinfer:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as flashinfer_prefill

            packed_conv = conv_parts[0] if len(conv_parts) == 1 else torch.cat(conv_parts, dim=0)
            query, key, value = self._split_qkv(packed_conv)
            context = get_context()
            offsets = context.qo_indptr
            slot_ids = context.state_slot_ids
            if offsets is None or slot_ids is None:
                raise RuntimeError("FlashInfer GDN prefill requires device metadata")
            with record_function("nanovllm::gdn_prefill_flashinfer"):
                core, _ = flashinfer_prefill(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    g=decay.contiguous(),
                    beta=beta.contiguous(),
                    scale=self.head_k_dim**-0.5,
                    initial_state=self.state_pool.recurrent_state,
                    output_state=self.state_pool.recurrent_state,
                    output_final_state=True,
                    cu_seqlens=offsets,
                    state_indices=slot_ids,
                    use_qk_l2norm_in_kernel=True,
                    use_cp=False,
                    backend="flashinfer",
                )
        else:
            core = torch.cat(core_parts, dim=0)
        return self._finalize_output(core, z)

    def _forward_decode_batch(
        self,
        hidden_states: torch.Tensor,
        state_slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.state_pool.allocated:
            raise RuntimeError("decode requires allocated GDN state slots")
        if state_slot_ids.ndim != 1 or state_slot_ids.numel() != hidden_states.size(0):
            raise RuntimeError("decode state_slot_ids must be 1D and match token batch size")

        slot_ids = state_slot_ids.to(device=hidden_states.device, dtype=torch.long)
        mixed_qkv, z, b, a = self._project_inputs(hidden_states)

        conv_state = self.state_pool.conv_state.index_select(0, slot_ids)
        next_conv_state = torch.cat(
            (conv_state[..., 1:], mixed_qkv.unsqueeze(-1).to(conv_state.dtype)), dim=-1
        )
        conv_output = F.conv1d(
            next_conv_state, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim
        )
        mixed_qkv = F.silu(conv_output.squeeze(-1)).to(hidden_states.dtype)
        self.state_pool.conv_state.index_copy_(0, slot_ids, next_conv_state)

        query, key, value = self._split_qkv(mixed_qkv)
        if self._can_use_flashinfer(hidden_states):
            from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

            with record_function("nanovllm::gdn_decode_flashinfer"):
                core, _ = gated_delta_rule_decode_pretranspose(
                    query.unsqueeze(1).contiguous(),
                    key.unsqueeze(1).contiguous(),
                    value.unsqueeze(1).contiguous(),
                    None,
                    self.A_log.float(),
                    a.unsqueeze(1).contiguous(),
                    self.dt_bias.float(),
                    b.unsqueeze(1).contiguous(),
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm=True,
                    initial_state=self.state_pool.recurrent_state,
                    initial_state_indices=state_slot_ids,
                    backend="flashinfer",
                )
            return self._finalize_output(core.squeeze(1), z)

        beta, decay = self._gates(b, a)
        query = l2norm(query.float()) * (self.head_k_dim**-0.5)
        key = l2norm(key.float())
        if self.num_v_heads != self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=1)
            key = key.repeat_interleave(repeats, dim=1)

        state = self.state_pool.recurrent_state.index_select(0, slot_ids)
        state = state * decay.exp()[..., None, None]
        kv_mem = (state * key.unsqueeze(-2)).sum(dim=-1)
        delta = (value.float() - kv_mem) * beta.unsqueeze(-1)
        state = state + (delta.unsqueeze(-1) * key.unsqueeze(-2))
        self.state_pool.recurrent_state.index_copy_(0, slot_ids, state)
        core = (state * query.unsqueeze(-2)).sum(dim=-1).to(hidden_states.dtype)
        return self._finalize_output(core, z)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            state_slots = context.state_slots
            state_prefix_lens = context.state_prefix_lens
            q_offsets = context.prefill_q_offsets
            if (
                state_slots is None
                or state_prefix_lens is None
                or q_offsets is None
                or len(state_slots) != len(state_prefix_lens)
                or len(q_offsets) != len(state_slots) + 1
                or q_offsets[0] != 0
                or q_offsets[-1] != hidden_states.size(0)
            ):
                raise RuntimeError("invalid or incomplete GDN prefill metadata")
            return self._forward_prefill_batch(
                hidden_states, state_slots, state_prefix_lens, q_offsets
            )
        if context.state_slot_ids is None:
            raise RuntimeError("decode requires device state_slot_ids")
        return self._forward_decode_batch(hidden_states, context.state_slot_ids)
