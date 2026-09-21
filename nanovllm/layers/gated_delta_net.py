import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class RMSNormGated(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Correctness-first Qwen3.5 Gated DeltaNet implementation.

    The active Conv/Recurrent states live in per-layer pools and are addressed
    directly with request-level state slots supplied by the runtime context.
    This removes any need for engine-side Gather/Scatter state copies.

    The runtime is intentionally Qwen3.5-MoE / TP=1 / eager-only. The state
    layout is kept dense so the Python recurrence can later be replaced by a
    fused kernel without changing scheduler ownership.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "linear_num_key_heads"
            )
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim,
            bias=False,
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
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

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )
        self.out_proj = nn.Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )

        self.register_buffer(
            "conv_state",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "recurrent_state",
            torch.empty(0),
            persistent=False,
        )

    def state_cache_nbytes(self, num_slots: int) -> int:
        conv_elements = (
            num_slots
            * self.conv_dim
            * self.conv_kernel_size
        )
        recurrent_elements = (
            num_slots
            * self.num_v_heads
            * self.head_k_dim
            * self.head_v_dim
        )
        conv_bytes = conv_elements * self.in_proj_qkv.weight.element_size()
        recurrent_bytes = recurrent_elements * torch.tensor(
            [],
            dtype=torch.float32,
        ).element_size()
        return conv_bytes + recurrent_bytes

    def allocate_state_cache(self, num_slots: int) -> None:
        device = self.in_proj_qkv.weight.device
        dtype = self.in_proj_qkv.weight.dtype
        self.conv_state = torch.zeros(
            num_slots,
            self.conv_dim,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.recurrent_state = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )

    def _validate_state_slot(self, slot_id: int) -> None:
        if not self.conv_state.numel() or not self.recurrent_state.numel():
            raise RuntimeError("GDN state cache is not allocated")
        if not 0 <= slot_id < self.conv_state.size(0):
            raise RuntimeError(f"invalid GDN state slot {slot_id}")

    def snapshot_state_slot(
        self,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy one committed request state to host memory.

        Conv state keeps the model dtype while the recurrent matrix keeps its
        FP32 correctness dtype. Snapshot compression is intentionally deferred.
        """
        self._validate_state_slot(slot_id)
        return (
            self.conv_state[slot_id].detach().cpu().clone(),
            self.recurrent_state[slot_id].detach().cpu().clone(),
        )

    def restore_state_slot(
        self,
        slot_id: int,
        snapshot: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._validate_state_slot(slot_id)
        conv_state, recurrent_state = snapshot
        expected_conv = self.conv_state[slot_id]
        expected_recurrent = self.recurrent_state[slot_id]
        if conv_state.shape != expected_conv.shape:
            raise RuntimeError(
                "GDN conv snapshot shape does not match active state slot"
            )
        if recurrent_state.shape != expected_recurrent.shape:
            raise RuntimeError(
                "GDN recurrent snapshot shape does not match active state slot"
            )
        expected_conv.copy_(conv_state)
        expected_recurrent.copy_(recurrent_state)

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        # Match HF causal_conv1d_update: keep K historical inputs, concatenate
        # the current chunk, convolve without padding, and retain the last T
        # outputs. For a fresh zero state this is equivalent to causal left
        # padding by K-1.
        seq_len = mixed_qkv.size(0)
        x = mixed_qkv.transpose(0, 1).unsqueeze(0)
        x_with_state = torch.cat(
            (conv_state.unsqueeze(0), x),
            dim=-1,
        ).to(self.conv1d.weight.dtype)
        conv_state.copy_(
            x_with_state[0, :, -self.conv_kernel_size :]
        )
        out = F.conv1d(
            x_with_state,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=0,
            groups=self.conv_dim,
        )
        out = out[:, :, -seq_len:]
        return F.silu(out).squeeze(0).transpose(0, 1).to(mixed_qkv.dtype)

    def _delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = query.dtype

        query = l2norm(query.float())
        key = l2norm(key.float())
        query = query * (self.head_k_dim ** -0.5)
        value = value.float()
        decay = decay.float()
        beta = beta.float()

        if self.num_v_heads != self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=1)
            key = key.repeat_interleave(repeats, dim=1)

        state = recurrent_state
        outputs = []
        for token_idx in range(query.size(0)):
            q_t = query[token_idx]
            k_t = key[token_idx]
            v_t = value[token_idx]

            state.mul_(decay[token_idx].exp()[..., None, None])
            kv_mem = (
                state * k_t.unsqueeze(-1)
            ).sum(dim=-2)
            delta = (
                v_t - kv_mem
            ) * beta[token_idx].unsqueeze(-1)
            state.add_(
                k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            )
            outputs.append(
                (state * q_t.unsqueeze(-1)).sum(dim=-2)
            )

        return torch.stack(outputs, dim=0).to(input_dtype)

    def _forward_sequence(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).view(
            -1,
            self.num_v_heads,
            self.head_v_dim,
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = self._causal_conv(
            mixed_qkv,
            conv_state,
        )
        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.view(
            -1,
            self.num_k_heads,
            self.head_k_dim,
        )
        key = key.view(
            -1,
            self.num_k_heads,
            self.head_k_dim,
        )
        value = value.view(
            -1,
            self.num_v_heads,
            self.head_v_dim,
        )

        beta = b.sigmoid()
        decay = (
            -self.A_log.float().exp()
            * F.softplus(a.float() + self.dt_bias)
        )
        core = self._delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            recurrent_state,
        )
        core = self.norm(
            core.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        return self.out_proj(
            core.view(-1, self.value_dim)
        )

    def _temporary_states(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state = torch.zeros(
            self.conv_dim,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        recurrent_state = torch.zeros(
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )
        return conv_state, recurrent_state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.state_slots is None:
            raise RuntimeError("GDN requires runtime state-slot metadata")
        if context.state_prefix_lens is None:
            raise RuntimeError(
                "GDN requires committed state-prefix metadata"
            )

        state_slots = context.state_slots
        state_prefix_lens = context.state_prefix_lens
        if len(state_slots) != len(state_prefix_lens):
            raise RuntimeError(
                "GDN state slot/prefix metadata must have equal length"
            )
        outputs = []

        if context.is_prefill:
            if context.prefill_q_offsets is None:
                raise RuntimeError(
                    "GDN prefill requires CPU packed query offsets"
                )
            q_offsets = context.prefill_q_offsets
            num_sequences = len(state_slots)
            if len(q_offsets) != num_sequences + 1:
                raise RuntimeError(
                    "packed query offsets must match state metadata"
                )
            if q_offsets[0] != 0:
                raise RuntimeError("packed query offsets must start at zero")
            if q_offsets[-1] != hidden_states.size(0):
                raise RuntimeError(
                    "packed query offsets do not cover hidden states"
                )

            for seq_idx, slot_id in enumerate(state_slots):
                q_start = q_offsets[seq_idx]
                q_end = q_offsets[seq_idx + 1]
                q_len = q_end - q_start
                expected_prefix = state_prefix_lens[seq_idx]

                if q_len <= 0:
                    raise RuntimeError(
                        f"invalid packed query length for sequence "
                        f"{seq_idx}: q={q_len}"
                    )
                if expected_prefix < 0:
                    raise RuntimeError(
                        f"invalid GDN state prefix for sequence "
                        f"{seq_idx}: {expected_prefix}"
                    )

                if self.conv_state.numel() and slot_id >= 0:
                    conv_state = self.conv_state[slot_id]
                    recurrent_state = self.recurrent_state[slot_id]
                    if expected_prefix == 0:
                        # Physical slots are reused across requests. A fresh
                        # request must discard stale state from the old owner.
                        conv_state.zero_()
                        recurrent_state.zero_()
                else:
                    if expected_prefix != 0:
                        raise RuntimeError(
                            "cached GDN prefix requires an allocated state slot"
                        )
                    conv_state, recurrent_state = self._temporary_states(
                        hidden_states.device,
                        hidden_states.dtype,
                    )

                outputs.append(
                    self._forward_sequence(
                        hidden_states[q_start:q_end],
                        conv_state,
                        recurrent_state,
                    )
                )
        else:
            if len(state_slots) != hidden_states.size(0):
                raise RuntimeError(
                    "decode state-slot count must match token batch size"
                )
            for token_idx, slot_id in enumerate(state_slots):
                if slot_id < 0 or not self.conv_state.numel():
                    raise RuntimeError(
                        "decode requires allocated GDN state slots"
                    )
                if state_prefix_lens[token_idx] < 0:
                    raise RuntimeError(
                        "decode state-prefix length must be non-negative"
                    )
                outputs.append(
                    self._forward_sequence(
                        hidden_states[token_idx : token_idx + 1],
                        self.conv_state[slot_id],
                        self.recurrent_state[slot_id],
                    )
                )

        return torch.cat(outputs, dim=0)
