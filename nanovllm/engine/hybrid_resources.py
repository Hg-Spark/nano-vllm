from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import GDNStateSnapshot, StateSlotManager


class HybridResources:
    """Reserve and release a request's KV blocks and GDN slot together."""

    def __init__(
        self,
        block_manager: BlockManager,
        state_manager: StateSlotManager,
    ):
        self.block_manager = block_manager
        self.state_manager = state_manager

    def validate_request(self, seq: Sequence) -> None:
        """Check that one request's logical history has matching resources."""
        if seq.block_table:
            self.state_manager.validate(seq)
        elif seq.state_slot >= 0:
            raise RuntimeError(f"sequence {seq.seq_id} owns state slot without KV allocation")
        elif seq.committed_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} has committed history without hybrid resources"
            )
        if seq.pending_state_snapshot is not None and seq.committed_tokens == 0:
            raise RuntimeError(
                f"sequence {seq.seq_id} has a pending snapshot without a cached prefix"
            )

    def reserve(self, seq: Sequence, target_tokens: int) -> None:
        old_num_blocks = len(seq.block_table)
        allocated_state = seq.state_slot < 0
        try:
            if allocated_state:
                self.state_manager.allocate(seq)
            self.block_manager.ensure_capacity(seq, target_tokens)
        except Exception:
            self.block_manager.truncate_blocks(seq, old_num_blocks)
            if allocated_state and seq.state_slot >= 0:
                self.state_manager.deallocate(seq)
            raise

    def restore_prefix(
        self,
        seq: Sequence,
        block_ids: tuple[int, ...],
        num_tokens: int,
        snapshot: GDNStateSnapshot,
    ) -> None:
        self.state_manager.allocate(seq)
        try:
            self.block_manager.attach_shared_prefix(
                seq,
                block_ids,
                num_tokens,
            )
        except Exception:
            self.state_manager.deallocate(seq)
            raise
        seq.committed_tokens = num_tokens
        seq.pending_state_snapshot = snapshot

    def release(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        self.state_manager.deallocate(seq)
