import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.registry import create_model
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


def _resolve_dtype(config) -> torch.dtype:
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype or torch.bfloat16


class ModelRunner:

    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.text_config
        self.block_size = config.kvcache_block_size

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        dtype = _resolve_dtype(self.hf_config)
        torch.set_default_dtype(dtype)
        torch.set_default_device("cuda")
        try:
            self.model = create_model(self.hf_config)
            load_model(self.model, config.model)
            self.sampler = Sampler()
            self.warmup_model()
            self.allocate_cache()
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

    def exit(self):
        torch.cuda.synchronize()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
            32,
        )
        if seq_len <= 0:
            raise ValueError("warmup sequence length must be positive")
        seq = Sequence([0] * seq_len)
        seq.num_scheduled_tokens = seq_len
        self.run([seq], True)
        torch.cuda.empty_cache()

    def _cache_modules(self):
        """Discover persistent-cache capabilities without model-specific paths."""
        kv_modules = []
        state_modules = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                kv_modules.append(module)
            if (
                callable(getattr(module, "state_cache_nbytes", None))
                and callable(getattr(module, "allocate_state_cache", None))
            ):
                state_modules.append(module)
        return kv_modules, state_modules

    def allocate_cache(self):
        config = self.config
        hf_config = self.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]

        kv_modules, state_modules = self._cache_modules()
        if not kv_modules:
            raise RuntimeError(
                "Qwen3.5-MoE config contains no full-attention layers"
            )
        if not state_modules:
            raise RuntimeError(
                "Qwen3.5-MoE config contains no GDN layers"
            )

        state_cache_bytes = sum(
            module.state_cache_nbytes(
                config.max_num_state_slots
            )
            for module in state_modules
        )
        model_param = next(self.model.parameters())
        block_bytes = (
            2
            * len(kv_modules)
            * self.block_size
            * hf_config.num_key_value_heads
            * hf_config.head_dim
            * model_param.element_size()
        )
        cache_budget = int(
            total * config.gpu_memory_utilization
            - used
            - peak
            + current
            - state_cache_bytes
        )
        config.num_kvcache_blocks = (
            cache_budget // block_bytes
        )
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                "insufficient GPU memory after reserving "
                "Qwen3.5-MoE recurrent state pools"
            )

        self.kv_cache = torch.empty(
            2,
            len(kv_modules),
            config.num_kvcache_blocks,
            self.block_size,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            device="cuda",
            dtype=model_param.dtype,
        )
        for layer_id, module in enumerate(kv_modules):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

        for module in state_modules:
            module.allocate_state_cache(
                config.max_num_state_slots
            )

    def prepare_block_tables(
        self,
        seqs: list[Sequence],
    ) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table
            + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        return torch.tensor(
            block_tables,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

    def prepare_prefill(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        q_offsets = [0]
        k_offsets = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        state_slots = []
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens
            state_slots.append(seq.state_slot)
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            q_offsets.append(q_offsets[-1] + seqlen_q)
            k_offsets.append(k_offsets[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            if not seq.block_table:
                continue

            start_block = start // self.block_size
            end_block = (
                end + self.block_size - 1
            ) // self.block_size
            for block_idx in range(
                start_block,
                end_block,
            ):
                slot_start = (
                    seq.block_table[block_idx]
                    * self.block_size
                )
                if block_idx == start_block:
                    slot_start += start % self.block_size
                if block_idx != end_block - 1:
                    slot_end = (
                        seq.block_table[block_idx]
                        * self.block_size
                        + self.block_size
                    )
                else:
                    slot_end = (
                        seq.block_table[block_idx]
                        * self.block_size
                        + end
                        - block_idx * self.block_size
                    )
                slot_mapping.extend(
                    range(slot_start, slot_end)
                )

        if k_offsets[-1] > q_offsets[-1]:
            block_tables = self.prepare_block_tables(seqs)

        cu_seqlens_q = torch.tensor(
            q_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            k_offsets,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(
            slot_mapping,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables,
            state_slots=tuple(state_slots),
            prefill_q_offsets=tuple(q_offsets),
            prefill_k_offsets=tuple(k_offsets),
        )

        return (
            torch.tensor(
                input_ids,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
            torch.tensor(
                positions,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
        )

    def prepare_decode(
        self,
        seqs: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = [seq.last_token for seq in seqs]
        positions = [len(seq) - 1 for seq in seqs]
        context_lens = [len(seq) for seq in seqs]
        state_slots = tuple(
            seq.state_slot for seq in seqs
        )
        slot_mapping = [
            seq.block_table[-1] * self.block_size
            + seq.last_block_num_tokens
            - 1
            for seq in seqs
        ]

        set_context(
            is_prefill=False,
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True),
            context_lens=torch.tensor(
                context_lens,
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True),
            block_tables=self.prepare_block_tables(seqs),
            state_slots=state_slots,
        )
        return (
            torch.tensor(
                input_ids,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
            torch.tensor(
                positions,
                dtype=torch.int64,
                pin_memory=True,
            ).cuda(non_blocking=True),
        )

    def _sample_indices(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        if not is_prefill:
            return list(range(len(seqs)))
        return [
            idx
            for idx, seq in enumerate(seqs)
            if (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
                == seq.num_tokens
            )
        ]

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
        )

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> list[int | None]:
        try:
            if is_prefill:
                input_ids, positions = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)

            hidden_states = self.run_model(
                input_ids,
                positions,
            )
            sample_indices = self._sample_indices(
                seqs,
                is_prefill,
            )
            results: list[int | None] = [
                None
                for _ in seqs
            ]
            if sample_indices:
                logits = self.model.compute_logits(
                    hidden_states,
                    sample_indices if is_prefill else None,
                )
                temperatures = torch.tensor(
                    [
                        seqs[idx].temperature
                        for idx in sample_indices
                    ],
                    dtype=torch.float32,
                    pin_memory=True,
                ).cuda(non_blocking=True)
                sampled = self.sampler(
                    logits,
                    temperatures,
                ).tolist()
                for idx, token_id in zip(
                    sample_indices,
                    sampled,
                ):
                    results[idx] = token_id
            return results
        finally:
            reset_context()
