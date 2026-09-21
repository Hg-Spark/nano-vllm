import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return x32 * torch.rsqrt((x32 * x32).sum(dim=-1, keepdim=True) + eps)


def causal_conv1d_stateful(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Readable causal depthwise Conv1d reference with explicit history.

    hidden_states: [T, C]
    conv_state:    [C, K - 1]
    weight:        [C, 1, K]
    """
    if hidden_states.ndim != 2 or conv_state.ndim != 2:
        raise ValueError("expected hidden_states=[T,C] and conv_state=[C,K-1]")
    x = hidden_states.transpose(0, 1).unsqueeze(0)
    history = conv_state.unsqueeze(0).to(x.dtype)
    x_with_history = torch.cat((history, x), dim=-1)
    out = F.conv1d(
        x_with_history.to(weight.dtype),
        weight,
        bias=None,
        padding=0,
        groups=hidden_states.shape[-1],
    )
    next_state = x_with_history[0, :, -(weight.shape[-1] - 1):]
    return F.silu(out.squeeze(0).transpose(0, 1)).to(hidden_states.dtype), next_state


def recurrent_gated_delta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_log: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent Gated DeltaNet scan.

    Shapes:
      query/key: [T, H, K]
      value:     [T, H, V]
      decay_log: [T, H] (<= 0)
      beta:      [T, H]
      state:     [H, K, V]

    The scan intentionally uses fp32 state/math. It is a correctness baseline,
    not the final high-throughput kernel.
    """
    input_dtype = query.dtype
    q = l2_normalize(query) / (query.shape[-1] ** 0.5)
    k = l2_normalize(key)
    v = value.float()
    g = decay_log.float()
    b = beta.float()
    state = recurrent_state.float()
    outputs = []

    for i in range(q.shape[0]):
        state = state * g[i].exp()[:, None, None]
        k_t = k[i]
        q_t = q[i]
        v_t = v[i]
        prediction = (state * k_t[:, :, None]).sum(dim=1)
        delta = (v_t - prediction) * b[i][:, None]
        state = state + k_t[:, :, None] * delta[:, None, :]
        outputs.append((state * q_t[:, :, None]).sum(dim=1))

    output = torch.stack(outputs, dim=0).to(input_dtype)
    return output, state


class GatedRMSNorm(torch.nn.Module):

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x.to(input_dtype) * self.weight
        x = x * F.silu(gate.float()).to(input_dtype)
        return x
