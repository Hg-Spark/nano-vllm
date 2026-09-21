from dataclasses import dataclass
import torch


def _resolve_dtype(value, default: torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.removeprefix("torch.")
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    return default


@dataclass(slots=True)
class GDNLayerState:
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor


class GDNStateManager:
    """Owns per-sequence, per-linear-layer GDN state.

    KV cache blocks are indexed by token positions and can be paged. GDN state is
    a fixed-size recurrent summary plus a short Conv1d history, so it has a
    different lifetime: allocate on first use, carry across chunked prefill and
    decode, reset on recomputation/preemption, release when the sequence ends.
    """

    def __init__(self, config, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.linear_layer_ids = {
            i for i, layer_type in enumerate(config.layer_types)
            if layer_type == "linear_attention"
        }
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.model_dtype = _resolve_dtype(
            getattr(config, "dtype", None),
            torch.get_default_dtype(),
        )
        self.state_dtype = _resolve_dtype(
            getattr(config, "mamba_ssm_dtype", None),
            torch.float32,
        )
        self._states: dict[int, dict[int, GDNLayerState]] = {}

    @property
    def bytes_per_sequence(self) -> int:
        conv_elems = self.conv_dim * (self.conv_kernel_size - 1)
        recurrent_elems = self.num_v_heads * self.head_k_dim * self.head_v_dim
        conv_bytes = conv_elems * torch.empty((), dtype=self.model_dtype).element_size()
        recurrent_bytes = recurrent_elems * torch.empty((), dtype=self.state_dtype).element_size()
        return len(self.linear_layer_ids) * (conv_bytes + recurrent_bytes)

    def get(self, seq_id: int, layer_idx: int) -> GDNLayerState:
        if layer_idx not in self.linear_layer_ids:
            raise KeyError(f"layer {layer_idx} is not a GDN layer")
        seq_states = self._states.setdefault(seq_id, {})
        state = seq_states.get(layer_idx)
        if state is None:
            state = GDNLayerState(
                conv_state=torch.zeros(
                    self.conv_dim,
                    self.conv_kernel_size - 1,
                    device=self.device,
                    dtype=self.model_dtype,
                ),
                recurrent_state=torch.zeros(
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                    device=self.device,
                    dtype=self.state_dtype,
                ),
            )
            seq_states[layer_idx] = state
        return state

    def reset_sequence(self, seq_id: int) -> None:
        self._states.pop(seq_id, None)

    def release_sequences(self, seq_ids: list[int] | tuple[int, ...]) -> None:
        for seq_id in seq_ids:
            self._states.pop(seq_id, None)

    def clear(self) -> None:
        self._states.clear()
