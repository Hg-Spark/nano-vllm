import torch
from torch import nn


GDN_SNAPSHOT_DTYPE = torch.bfloat16


class GDNStatePool(nn.Module):
    """Physical Conv/Recurrent state slots for one GDN layer."""

    def __init__(
        self,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
    ):
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
        return bool(
            self.conv_state.numel()
            and self.recurrent_state.numel()
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
            self.conv_kernel_size,
            device=device,
            dtype=self.conv_state.dtype,
        )
        self.recurrent_state = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=torch.float32,
        )

    def _validate_slot(self, slot_id: int) -> None:
        if not self.allocated:
            raise RuntimeError("GDN state cache is not allocated")
        if not 0 <= slot_id < self.conv_state.size(0):
            raise RuntimeError(f"invalid GDN state slot {slot_id}")

    def slot(
        self,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_slot(slot_id)
        return (
            self.conv_state[slot_id],
            self.recurrent_state[slot_id],
        )

    def clear(self, slot_id: int) -> None:
        conv_state, recurrent_state = self.slot(slot_id)
        conv_state.zero_()
        recurrent_state.zero_()

    def temporary(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(
                self.conv_dim,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=torch.float32,
            ),
        )

    def snapshot_state_slot(
        self,
        slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_state, recurrent_state = self.slot(slot_id)
        return (
            conv_state.detach().to(
                device="cpu",
                dtype=GDN_SNAPSHOT_DTYPE,
                copy=True,
            ),
            recurrent_state.detach().to(
                device="cpu",
                dtype=GDN_SNAPSHOT_DTYPE,
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
        if conv_state.shape != expected_conv.shape:
            raise RuntimeError(
                "GDN conv snapshot shape does not match active state slot"
            )
        if recurrent_state.shape != expected_recurrent.shape:
            raise RuntimeError(
                "GDN recurrent snapshot shape does not match active state slot"
            )
        if conv_state.dtype != GDN_SNAPSHOT_DTYPE:
            raise RuntimeError("GDN conv snapshot must use BF16 storage")
        if recurrent_state.dtype != GDN_SNAPSHOT_DTYPE:
            raise RuntimeError("GDN recurrent snapshot must use BF16 storage")
        expected_conv.copy_(conv_state)
        expected_recurrent.copy_(recurrent_state)
