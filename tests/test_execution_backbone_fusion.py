"""Independent oracles for full-precision pair layout and adaLN grouping."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from test_g1_integration import perturb

from src.models.execution import configure_execution
from src.models.execution.ops import (
    normalize_pairs,
    project_normalized_pairs,
    project_pair_bias_group,
    project_pair_heads,
    project_time_modulations,
)
from src.models.simple.attention import AttentionPairBias
from src.models.simple.transformers import DiT


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def assert_derivatives(a, b, variables, atol=1e-10, rtol=1e-10):
    gradient = torch.randn_like(a)
    left = torch.autograd.grad(a, variables, gradient, retain_graph=True)
    right = torch.autograd.grad(b, variables, gradient)
    for old, new in zip(left, right):
        torch.testing.assert_close(old, new, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(2, 3, 5, 7), (1, 1, 1, 7), (0, 3, 5, 7)])
def test_head_first_projection_matches_linear_and_parameter_gradients(shape):
    torch.manual_seed(19)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    layer = torch.nn.Linear(7, 4).double()
    old = layer(x).permute(0, 3, 1, 2).contiguous()
    new = project_pair_heads(x, layer.weight, layer.bias)
    assert new.is_contiguous()
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    assert_derivatives(old, new, (x, *layer.parameters()))


def test_head_first_group_projection_preserves_independent_affines():
    torch.manual_seed(31)
    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
    norms = [torch.nn.LayerNorm(7, elementwise_affine=i != 1).double() for i in range(3)]
    layers = [torch.nn.Linear(7, i + 2, bias=i != 1).double() for i in range(3)]
    for module in norms + layers:
        perturb(module)
    old = torch.cat([layer(norm(x)) for norm, layer in zip(norms, layers)], -1)
    new = torch.cat(
        project_pair_bias_group(normalize_pairs(x, norms[0]), zip(norms, layers), head_first=True),
        1,
    ).permute(0, 2, 3, 1)
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    assert_derivatives(old, new, (x, *(p for m in norms + layers for p in m.parameters())))


@pytest.mark.parametrize("impl", ["manual", "pytorch"])
@pytest.mark.parametrize("pair_mode", ["raw", "normalized", "projected"])
def test_attention_layout_paths_match_outputs_gradients_and_keep_contiguous_mask(impl, pair_mode):
    torch.manual_seed(17)
    old = AttentionPairBias(16, 7, 4, attention_impl=impl)
    perturb(old)
    new = deepcopy(old)
    new.head_first_pair_bias = True
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
    inputs = [
        torch.randn(2, 5, 16, requires_grad=True),
        torch.randn(2, 5, 5, 7, requires_grad=True),
    ]
    clones = [x.detach().clone().requires_grad_() for x in inputs]

    def apply(module, values):
        x, z = values
        normalized = pair_mode != "raw"
        projected = pair_mode == "projected"
        if normalized:
            z = normalize_pairs(z, module.proj_z[0])
        if projected:
            z = project_normalized_pairs(
                z, module.proj_z[0], module.proj_z[1], head_first=module.head_first_pair_bias
            )
        kwargs = dict(
            pair_normalized=normalized,
            pair_projected=projected,
            pair_head_first=projected and module.head_first_pair_bias,
        )
        bias = module._create_attn_mask(mask, z, **kwargs)
        assert bias.is_contiguous()
        assert bias.shape == (2, 4, 5, 5)
        return module(x, mask, z, **kwargs)

    a, b = apply(old, inputs), apply(new, clones)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=5e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip([*inputs, *old.parameters()], [*clones, *new.parameters()]):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-4)
    assert old.state_dict().keys() == new.state_dict().keys()


def test_time_projection_group_matches_independent_silu_and_parameter_gradients():
    torch.manual_seed(37)
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    layers = [torch.nn.Linear(8, 48).double() for _ in range(5)]
    old = torch.cat([layer(F.silu(x)) for layer in layers], -1)
    new = torch.cat(project_time_modulations(F.silu(x), layers), -1)
    torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
    assert_derivatives(old, new, (x, *(p for m in layers for p in m.parameters())))


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("time_chunk", [2, 4, 16])
def test_dit_grouping_preserves_hooks_masks_gradients_keys(checkpoint, pairs, time_chunk):
    torch.manual_seed(47)
    old = DiT(
        depth=5,
        heads=2,
        dim=8,
        dim_pair=7 if pairs else None,
        attention_impl="pytorch",
        activation_checkpointing=checkpoint,
    )
    perturb(old)
    new = deepcopy(old)
    new.pair_norm_reuse = True
    new.pair_bias_chunk_size = 4
    new.time_modulation_chunk_size = time_chunk
    for layer in new.layers:
        layer.attn.head_first_pair_bias = True
        layer.attn.broadcast_masks = True
    # Mixed eps/inf values and a mixed head-layout chunk must stay correct.
    for module in (old, new):
        module.layers[1].attn.inf = 1e4
        if pairs:
            module.layers[2].attn.proj_z[0].eps = 1e-3
    new.layers[2].attn.head_first_pair_bias = False
    xs = [torch.randn(2, 5, 8, requires_grad=True), torch.rand(2, requires_grad=True)]
    if pairs:
        xs.append(torch.randn(2, 5, 5, 7, requires_grad=True))
    ys = [x.detach().clone().requires_grad_() for x in xs]
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
    traces = [[], []]

    def hook(trace):
        def update(index, x):
            trace.append(index)
            return x * (1 + index * 0.01)

        return update

    a = old(xs[0], xs[1], mask, xs[2] if pairs else None, after_layer=hook(traces[0]))
    b = new(ys[0], ys[1], mask, ys[2] if pairs else None, after_layer=hook(traces[1]))
    torch.testing.assert_close(a, b, atol=2e-6, rtol=5e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip([*xs, *old.parameters()], [*ys, *new.parameters()]):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=5e-4)
    assert traces[0] == traces[1] == [1, 2, 3, 4, 5]
    assert old.state_dict().keys() == new.state_dict().keys()
    new.load_state_dict(old.state_dict(), strict=True)


def test_compiled_blocks_accept_precomputed_bias_and_modulation_then_reset():
    torch.manual_seed(61)
    old = DiT(depth=2, heads=2, dim=8, dim_pair=7, attention_impl="pytorch")
    perturb(old)
    new = deepcopy(old)
    configure_execution(
        new,
        dict(
            pair_norm_reuse=True,
            pair_bias_chunk_size=4,
            head_first_pair_bias=True,
            time_modulation_chunk_size=4,
            broadcast_masks=True,
            compile_blocks=True,
            compile_backend="aot_eager",
            compile_dynamic=True,
        ),
    )
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    z = torch.randn(2, 4, 4, 7, requires_grad=True)
    w = z.detach().clone().requires_grad_()
    t, mask = torch.rand(2), torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]])
    a, b = old(x, t, mask, z), new(y, t, mask, w)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=5e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip([x, z, *old.parameters()], [y, w, *new.parameters()]):
        torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=5e-4)
    assert old.state_dict().keys() == new.state_dict().keys()
    configure_execution(new, {})
    assert new.time_modulation_chunk_size == 1
    assert not new.layers[0].attn.head_first_pair_bias
    assert new.layers[0]._compiled_call_impl is None
