import torch

from nanovllm.layers.gated_delta import causal_conv1d_stateful, recurrent_gated_delta


def test_causal_conv_state_matches_single_pass():
    torch.manual_seed(0)
    tokens, channels, kernel = 7, 5, 4
    x = torch.randn(tokens, channels)
    weight = torch.randn(channels, 1, kernel)
    initial = torch.zeros(channels, kernel - 1)

    full, full_state = causal_conv1d_stateful(x, initial.clone(), weight)

    first, state = causal_conv1d_stateful(x[:3], initial.clone(), weight)
    second, state = causal_conv1d_stateful(x[3:], state, weight)

    torch.testing.assert_close(torch.cat((first, second)), full)
    torch.testing.assert_close(state, full_state)


def test_recurrent_scan_matches_chunked_execution():
    torch.manual_seed(1)
    tokens, heads, key_dim, value_dim = 8, 4, 3, 5
    q = torch.randn(tokens, heads, key_dim)
    k = torch.randn(tokens, heads, key_dim)
    v = torch.randn(tokens, heads, value_dim)
    g = -torch.rand(tokens, heads)
    beta = torch.sigmoid(torch.randn(tokens, heads))
    initial = torch.zeros(heads, key_dim, value_dim)

    full, full_state = recurrent_gated_delta(q, k, v, g, beta, initial.clone())

    first, state = recurrent_gated_delta(
        q[:5], k[:5], v[:5], g[:5], beta[:5], initial.clone()
    )
    second, state = recurrent_gated_delta(
        q[5:], k[5:], v[5:], g[5:], beta[5:], state
    )

    torch.testing.assert_close(torch.cat((first, second)), full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, full_state, rtol=1e-5, atol=1e-5)
