"""Internal script adapters. Keep scheduler/model structure access in one place.

These functions use existing runtime operations, not a second scheduler. The
caller owns the engine and must close it on failure; probes never retry a partial
experiment. Runtime imports are local to keep script help usable without CUDA.
"""


def prepare_requests(engine, prompts, sampling):
    if not engine.is_finished():
        raise RuntimeError("benchmark preparation requires an idle engine")
    return [engine._prepare_request(prompt, sampling) for prompt in prompts]


def enqueue_sequences(engine, sequences):
    if not engine.is_finished():
        raise RuntimeError("benchmark enqueue requires an idle engine")
    for seq in sequences:
        engine.scheduler.add(seq)


def vocabulary_size(engine):
    return engine.model_runner.config.text_config.vocab_size


def prepare_fixed_decode_batch(engine, prompt, sampling, batch_size):
    if not engine.is_finished():
        raise RuntimeError("fixed-batch setup requires an idle engine")
    # Park completed prefills so subsequent setup does not decode them.
    # Setup failures abort profiling; the caller closes the engine.
    parked = []
    for _ in range(batch_size):
        engine.add_request(prompt, sampling)
        while engine.scheduler.waiting:
            scheduled = engine.scheduler.schedule()
            if scheduled.decode_chunks:
                raise RuntimeError("fixed-batch setup unexpectedly scheduled decode work")
            engine._run_batch(scheduled.prefill_chunks, True)
        if len(engine.scheduler.running) != 1:
            raise RuntimeError("fixed-batch setup expected exactly one completed prefill")
        parked.append(engine.scheduler.running.pop())
    engine.scheduler.running.extend(parked)


def find_prefix_metadata(engine, prompt):
    from nanovllm.engine.sequence import Sequence

    entry = engine.scheduler.block_manager.find_prefix(Sequence(prompt), max_tokens=len(prompt) - 1)
    if entry is None:
        return None
    return {
        "num_tokens": entry.num_tokens,
        "snapshot_bytes": sum(
            tensor.numel() * tensor.element_size()
            for layer in entry.state_snapshot.layers
            for tensor in layer
        ),
    }
