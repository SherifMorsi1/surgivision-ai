"""Behavioural tests for the temporal model, including strict causality."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from surgivision.models.tcn import MultiStageTCN, SingleStageTCN  # noqa: E402


def _model(causal: bool, stages: int = 2, layers: int = 4) -> MultiStageTCN:
    torch.manual_seed(0)
    model = MultiStageTCN(
        num_stages=stages,
        num_layers=layers,
        num_features=16,
        in_channels=32,
        num_classes=7,
        causal=causal,
    )
    return model.eval()


def test_output_shape_is_stages_batch_classes_time():
    model = _model(causal=False)
    out = model(torch.randn(2, 32, 50))
    assert out.shape == (2, 2, 7, 50)


def test_receptive_field_grows_exponentially_with_depth():
    stage = SingleStageTCN(
        num_layers=10, in_channels=32, num_features=16, num_classes=7
    )
    # dilations 1,2,4,...,512 sum to 1023
    assert stage.receptive_field == 1 + 2 * 1023


def test_causal_model_cannot_see_the_future():
    """Perturbing frame t must leave every output before t untouched."""

    model = _model(causal=True)
    x = torch.randn(1, 32, 40)
    perturbed = x.clone()
    perturbed[0, :, 30] += 10.0

    with torch.no_grad():
        base = model(x)
        after = model(perturbed)

    past = slice(0, 30)
    assert torch.allclose(base[..., past], after[..., past], atol=1e-6), (
        "a causal model leaked information from a future frame into the past"
    )


def test_causal_model_still_responds_at_and_after_the_change():
    model = _model(causal=True)
    x = torch.randn(1, 32, 40)
    perturbed = x.clone()
    perturbed[0, :, 30] += 10.0

    with torch.no_grad():
        base = model(x)
        after = model(perturbed)

    assert not torch.allclose(base[..., 30:], after[..., 30:], atol=1e-6)


def test_non_causal_model_does_use_future_context():
    """Confirms the causality test above is actually discriminating."""

    model = _model(causal=False)
    x = torch.randn(1, 32, 40)
    perturbed = x.clone()
    perturbed[0, :, 30] += 10.0

    with torch.no_grad():
        base = model(x)
        after = model(perturbed)

    assert not torch.allclose(base[..., :30], after[..., :30], atol=1e-6)


def test_length_is_preserved_for_odd_and_even_sequences():
    model = _model(causal=True)
    for length in (17, 18, 129):
        assert model(torch.randn(1, 32, length)).shape[-1] == length


def test_gradients_reach_the_first_stage():
    model = _model(causal=False).train()
    out = model(torch.randn(1, 32, 32))
    out.sum().backward()
    grad = model.stage1.conv_in.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_mask_zeroes_padded_positions():
    model = _model(causal=False)
    x = torch.randn(1, 32, 20)
    mask = torch.ones(1, 1, 20)
    mask[..., 15:] = 0.0
    with torch.no_grad():
        out = model(x, mask)
    assert torch.allclose(out[..., 15:], torch.zeros_like(out[..., 15:]))


def test_rejects_non_3d_input():
    model = _model(causal=False)
    with pytest.raises(ValueError, match="batch, channels, time"):
        model(torch.randn(32, 50))


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="at least one layer"):
        SingleStageTCN(num_layers=0, in_channels=32, num_features=16, num_classes=7)
    with pytest.raises(ValueError, match="(?i)at least one stage"):
        MultiStageTCN(num_stages=0)
