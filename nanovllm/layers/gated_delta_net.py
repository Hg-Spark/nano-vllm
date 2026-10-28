import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from nanovllm.layers.gdn_state import GDNStatePool
from nanovllm.utils.context import get_context


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


def _align_flashinfer_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    return x if x.data_ptr() % 16 == 0 else x.clone()


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
    """FlashInfer-only Qwen3.5 Gated DeltaNet."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.value_dim + 2 * self.num_v_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.state_pool = GDNStatePool(
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )

    def _causal_conv(self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        from causal_conv1d import causal_conv1d_fn

        # Final states require channel-last input and initial-state layouts.
        x = mixed_qkv.contiguous().transpose(0, 1).unsqueeze(0)
        initial_states = conv_state.transpose(0, 1).contiguous().transpose(0, 1)
        out, final_states = causal_conv1d_fn(
            x,
            self.conv1d.weight.squeeze(1),
            None,
            initial_states=initial_states.unsqueeze(0),
            return_final_states=True,
            activation="silu",
        )
        conv_state.copy_(final_states[0])
        return out.squeeze(0).transpose(0, 1)

    def _causal_conv_batch(
        self,
        mixed_qkv: torch.Tensor,
        conv_states: list[torch.Tensor],
        q_offsets: tuple[int, ...],
    ) -> torch.Tensor:
        if len(conv_states) == 1:
            return self._causal_conv(mixed_qkv, conv_states[0])

        from causal_conv1d import causal_conv1d_fn

        lengths = [q_offsets[index + 1] - q_offsets[index] for index in range(len(conv_states))]
        if len(set(lengths)) == 1:
            # Use the kernel's native batched initial/final states for equal-size
            # chunks. Unlike the packed fallback, this does not materialize
            # history tokens in the convolution input or repack its output.
            x = mixed_qkv.contiguous().view(len(conv_states), lengths[0], self.conv_dim)
            initial_states = torch.stack([state.T for state in conv_states]).transpose(1, 2)
            out, final_states = causal_conv1d_fn(
                x.transpose(1, 2),
                self.conv1d.weight.squeeze(1),
                None,
                initial_states=initial_states,
                return_final_states=True,
                activation="silu",
            )
            for state, final_state in zip(conv_states, final_states):
                state.copy_(final_state)
            return out.transpose(1, 2).reshape(-1, self.conv_dim)

        # Variable-size chunks retain the packed single-kernel fallback.
        # Explicit history windows isolate their convolution state.
        history = self.conv_kernel_size - 1
        segments = []
        for index, state in enumerate(conv_states):
            segments.extend((state.T, mixed_qkv[q_offsets[index] : q_offsets[index + 1]]))
        packed = torch.cat(segments, dim=0)
        output = (
            causal_conv1d_fn(
                packed.T.unsqueeze(0),
                self.conv1d.weight.squeeze(1),
                activation="silu",
            )
            .squeeze(0)
            .T
        )

        parts = []
        cursor = 0
        for index, state in enumerate(conv_states):
            start = cursor + history
            end = start + q_offsets[index + 1] - q_offsets[index]
            parts.append(output[start:end])
            # Convolution history stores raw projected inputs, not SiLU outputs.
            state.copy_(packed[end - history : end].T)
            cursor = end
        return torch.cat(parts, dim=0)

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
        offsets: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as flashinfer_prefill

        mixed_qkv, z, b, a = self._project_inputs(hidden_states)
        beta, decay = self._gates(b, a)
        conv_states = []
        for seq_idx, slot_id in enumerate(state_slots):
            conv_state, recurrent_state = self.state_pool.slot(slot_id)
            if state_prefix_lens[seq_idx] == 0:
                conv_state.zero_()
                recurrent_state.zero_()
            conv_states.append(conv_state)

        packed_conv = self._causal_conv_batch(mixed_qkv, conv_states, q_offsets)
        query, key, value = self._split_qkv(packed_conv)
        # FlashInfer 0.7.0's prefill path ignores its Q/K normalization flag.
        query = _l2norm(query.float()).to(query.dtype).contiguous()
        key = _l2norm(key.float()).to(key.dtype).contiguous()
        with record_function("nanovllm::gdn_prefill_flashinfer"):
            core, _ = flashinfer_prefill(
                query,
                key,
                value.contiguous(),
                g=decay.exp(),
                beta=beta,
                scale=self.head_k_dim**-0.5,
                initial_state=self.state_pool.recurrent_state,
                output_state=self.state_pool.recurrent_state,
                output_final_state=True,
                cu_seqlens=offsets,
                state_indices=slot_ids,
                use_qk_l2norm_in_kernel=False,
                use_cp=False,
                backend="flashinfer",
            )
        return self._finalize_output(core, z)

    def _forward_decode_batch(
        self,
        hidden_states: torch.Tensor,
        state_slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        from causal_conv1d import causal_conv1d_update
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        mixed_qkv, z, b, a = self._project_inputs(hidden_states)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            self.state_pool.conv_state,
            self.conv1d.weight.squeeze(1),
            None,
            activation="silu",
            conv_state_indices=state_slot_ids,
        )
        query, key, value = self._split_qkv(mixed_qkv)
        a = _align_flashinfer_tensor(a.unsqueeze(1))
        b = _align_flashinfer_tensor(b.unsqueeze(1))
        aligned_slot_ids = _align_flashinfer_tensor(state_slot_ids.to(torch.int32))
        with record_function("nanovllm::gdn_decode_flashinfer"):
            core, _ = gated_delta_rule_decode_pretranspose(
                query.unsqueeze(1).contiguous(),
                key.unsqueeze(1).contiguous(),
                value.unsqueeze(1).contiguous(),
                None,
                self.A_log.detach().float(),
                a,
                self.dt_bias.detach().float(),
                b,
                scale=self.head_k_dim**-0.5,
                use_qk_l2norm=True,
                initial_state=self.state_pool.recurrent_state,
                initial_state_indices=aligned_slot_ids,
                backend="flashinfer",
            )
        return self._finalize_output(core.squeeze(1).to(hidden_states.dtype), z)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            state_slots = context.state_slots
            state_prefix_lens = context.state_prefix_lens
            q_offsets = context.prefill_q_offsets
            return self._forward_prefill_batch(
                hidden_states,
                state_slots,
                state_prefix_lens,
                q_offsets,
                context.qo_indptr,
                context.state_slot_ids,
            )
        return self._forward_decode_batch(hidden_states, context.state_slot_ids)
