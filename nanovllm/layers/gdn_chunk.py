"""Blockwise FP32 gated delta recurrence for Qwen3.5 prefill.

The recurrent equation, rather than another engine's implementation, defines
this computation. For each position t, with normalized k and q:

    M_t = exp(g_t) M_(t-1) + k_t [beta_t (v_t - k_t^T exp(g_t) M_(t-1))]^T
    o_t = q_t^T M_t / sqrt(key_dim)

Within a block, causal key interactions form a unit-lower-triangular system.
Solving for the innovations lets us compute outputs and the final state using
matrix multiplications. The final, possibly short, block needs no padding.
The input/output state layout is [batch, heads, key_dim, value_dim].
"""

import torch


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_dtype = query.dtype

    # Group the time axis inside each head; accumulate the recurrent state
    # in FP32 independently of the activation dtype.
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    log_gate = decay.transpose(1, 2).float()
    update_rate = beta.transpose(1, 2).float()

    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (q.size(-1) ** -0.5)

    state = initial_state.float().clone()
    output_blocks = []
    for begin in range(0, q.size(2), 64):
        end = min(begin + 64, q.size(2))
        block_q = q[:, :, begin:end]
        block_k = k[:, :, begin:end]
        block_v = v[:, :, begin:end]
        block_beta = update_rate[:, :, begin:end]
        prefix_log_gate = log_gate[:, :, begin:end].cumsum(dim=-1)

        # Discount from an earlier token j to the current token i.
        # Qwen3.5 log gates are nonpositive, so the causal values are <= 1.
        elapsed = prefix_log_gate.unsqueeze(-1) - prefix_log_gate.unsqueeze(-2)
        discount = elapsed.clamp(max=0).exp()

        # Solve the causal delta-correction equations together, instead of
        # iterating over tokens in Python.
        key_similarity = block_k @ block_k.transpose(-1, -2)
        coupling = (block_beta.unsqueeze(-1) * key_similarity * discount).tril(-1)
        identity = torch.eye(end - begin, device=q.device, dtype=q.dtype)
        innovation = torch.linalg.solve_triangular(
            coupling + identity,
            block_beta.unsqueeze(-1)
            * (block_v - (block_k @ state) * prefix_log_gate.exp().unsqueeze(-1)),
            upper=False,
        )

        initial_read = (block_q @ state) * prefix_log_gate.exp().unsqueeze(-1)
        corrected_read = ((block_q @ block_k.transpose(-1, -2)) * discount).tril() @ innovation
        output_blocks.append(initial_read + corrected_read)

        # Advance the incoming [K,V] state past this block.
        block_log_gate = prefix_log_gate[..., -1]
        innovation_to_end = (block_log_gate.unsqueeze(-1) - prefix_log_gate).exp()
        state = state * block_log_gate.exp().unsqueeze(-1).unsqueeze(-1) + block_k.transpose(
            -1, -2
        ) @ (innovation * innovation_to_end.unsqueeze(-1))

    output = torch.cat(output_blocks, dim=2).transpose(1, 2)
    return output.to(output_dtype).contiguous(), state
