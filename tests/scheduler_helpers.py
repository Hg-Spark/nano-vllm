from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    max_num_batched_tokens=4,
    max_num_seqs=4,
    num_blocks=16,
    block_size=16,
):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_ids=(99, 100),
        kvcache_block_size=block_size,
        max_prefix_cache_entries=16,
    )
    return Scheduler(config, num_blocks)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids)
    scheduler.block_manager.ensure_capacity(seq, len(seq))
    scheduler.state_manager.allocate(seq)
    seq.committed_tokens = len(seq) - 1
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


def scheduled_sequences(chunks):
    return [chunk.seq for chunk in chunks]
