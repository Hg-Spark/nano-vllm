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
            raise RuntimeError(
                f"sequence {seq.seq_id} owns state slot "
                "without KV allocation"
            )
        elif seq.committed_tokens:
            raise RuntimeError(
                f"sequence {seq.seq_id} has committed history "
                "without hybrid resources"
            )
        if (
            seq.pending_state_snapshot is not None
            and seq.committed_tokens == 0
        ):
            raise RuntimeError(
                f"sequence {seq.seq_id} has a pending snapshot "
                "without a cached prefix"
            )

    def validate_accounting(self, seqs: list[Sequence]) -> None:
        """Verify KV refs and GDN slots match request/cache ownership."""
        seq_ids = [seq.seq_id for seq in seqs]
        if len(seq_ids) != len(set(seq_ids)):
            raise RuntimeError("sequence appears in multiple scheduler queues")

        expected_refs = [0] * len(self.block_manager.block_refcounts)
        for seq in seqs:
            self.validate_request(seq)
            if len(seq.block_table) != len(set(seq.block_table)):
                raise RuntimeError(
                    f"sequence {seq.seq_id} contains duplicate KV blocks"
                )
            for block_id in seq.block_table:
                if not 0 <= block_id < len(expected_refs):
                    raise RuntimeError(f"invalid KV block {block_id}")
                expected_refs[block_id] += 1

        for entry in self.block_manager.prefix_entries():
            for block_id in entry.block_ids:
                if not 0 <= block_id < len(expected_refs):
                    raise RuntimeError(f"invalid cached KV block {block_id}")
                expected_refs[block_id] += 1

        if expected_refs != self.block_manager.block_refcounts:
            raise RuntimeError("KV cache refcount accounting mismatch")

        free_blocks = list(self.block_manager.free_block_ids)
        if len(free_blocks) != len(set(free_blocks)):
            raise RuntimeError("duplicate KV block in free list")
        expected_free_blocks = {
            block_id
            for block_id, refcount in enumerate(expected_refs)
            if refcount == 0
        }
        if set(free_blocks) != expected_free_blocks:
            raise RuntimeError("KV cache free-list accounting mismatch")

        expected_owners = [None] * len(self.state_manager.slot_owners)
        for seq in seqs:
            if seq.state_slot < 0:
                continue
            if not 0 <= seq.state_slot < len(expected_owners):
                raise RuntimeError(f"invalid state slot {seq.state_slot}")
            if expected_owners[seq.state_slot] is not None:
                raise RuntimeError(
                    f"state slot {seq.state_slot} has multiple owners"
                )
            expected_owners[seq.state_slot] = seq.seq_id

        if expected_owners != self.state_manager.slot_owners:
            raise RuntimeError("GDN state-slot accounting mismatch")

        free_slots = list(self.state_manager.free_slot_ids)
        if len(free_slots) != len(set(free_slots)):
            raise RuntimeError("duplicate GDN state slot in free list")
        expected_free_slots = {
            slot_id
            for slot_id, owner in enumerate(expected_owners)
            if owner is None
        }
        if set(free_slots) != expected_free_slots:
            raise RuntimeError("GDN state free-list accounting mismatch")

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
