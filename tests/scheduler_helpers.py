from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(
    max_num_batched_tokens=4,
    max_num_seqs=4,
    num_blocks=16,
    block_size=16,
    max_prefix_cache_entries=16,
    max_prefix_cache_snapshot_bytes=None,
    snapshot_nbytes=0,
    prefix_cache_eager_snapshot=True,
    max_gdn_checkpoints=4,
    eos_token_ids=(99, 100),
):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos_token_ids=eos_token_ids,
        kv_cache_block_size=block_size,
        max_prefix_cache_entries=max_prefix_cache_entries,
        max_prefix_cache_snapshot_bytes=max_prefix_cache_snapshot_bytes,
        prefix_cache_eager_snapshot=prefix_cache_eager_snapshot,
        max_gdn_checkpoints=max_gdn_checkpoints,
    )
    return Scheduler(config, num_blocks, snapshot_nbytes=snapshot_nbytes)


def make_running_sequence(scheduler, token_ids):
    seq = Sequence(token_ids, SamplingParams(max_tokens=2))
    scheduler.block_manager.ensure_capacity(seq, len(seq))
    scheduler.state_manager.allocate(seq)
    seq.committed_tokens = len(seq) - 1
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


def scheduled_sequences(chunks):
    return [chunk.seq for chunk in chunks]
