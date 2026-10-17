import torch
from torch import nn


class GDNStatePool(nn.Module):
    """Physical Conv/Recurrent state slots for one GDN layer.

    Recurrent state uses the canonical ``[H, V, K]`` layout.
    """

    def __init__(
        self,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

        # These zero-sized buffers act as device/dtype anchors before the
        # runtime knows how many active request slots can fit.
        self.register_buffer(
            "conv_state",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "recurrent_state",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

    @property
    def allocated(self) -> bool:
        return bool(self.conv_state.numel() and self.recurrent_state.numel())

    def state_cache_nbytes(self, num_slots: int) -> int:
        conv_elements = num_slots * self.conv_dim * (self.conv_kernel_size - 1)
        recurrent_elements = num_slots * self.num_v_heads * self.head_k_dim * self.head_v_dim
        return (
            conv_elements * self.conv_state.element_size()
            + recurrent_elements * self.recurrent_state.element_size()
        )

    def allocate_state_cache(self, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        device = self.conv_state.device
        self.conv_state = torch.zeros(
            num_slots,
            self.conv_dim,
            self.conv_kernel_size - 1,
            device=device,
            dtype=self.conv_state.dtype,
        )
        self.recurrent_state = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=torch.float32,
        )

    def slot(
        self,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.conv_state[slot_id],
            self.recurrent_state[slot_id],
        )

    def snapshot_state_slot(
        self,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state, recurrent_state = self.slot(slot_id)
        return (
            conv_state.detach().to(
                device="cpu",
                copy=True,
            ),
            recurrent_state.detach().to(
                device="cpu",
                copy=True,
            ),
        )

    def restore_state_slot(
        self,
        slot_id: int,
        snapshot: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        expected_conv, expected_recurrent = self.slot(slot_id)
        conv_state, recurrent_state = snapshot
        expected_conv.copy_(conv_state)
        expected_recurrent.copy_(recurrent_state)
