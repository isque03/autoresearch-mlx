"""Regression test for the exact bug class behind nanochat's real SFT NaN
crash: an all-masked batch (every target is padding/ignore_index) causes a
0/0 in a naive mean-over-valid-tokens loss. train.py's GPT.__call__ already
guards this via mx.maximum(mx.sum(valid), 1) -- this test locks that
guarantee in explicitly so it can't regress silently."""
import math

import mlx.core as mx

from train import GPT, GPTConfig


def tiny_model():
    config = GPTConfig(sequence_len=64, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    return model


def test_all_masked_targets_never_produce_nan_or_inf():
    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    all_masked_targets = mx.full((1, 8), -1)
    loss = model(tokens, targets=all_masked_targets)
    value = float(loss.item())
    assert math.isfinite(value), f"expected a finite loss on an all-masked batch, got {value}"


def test_partially_masked_targets_still_finite():
    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = mx.array([[-1, -1, -1, -1, -1, -1, -1, 5]])
    loss = model(tokens, targets=targets)
    assert math.isfinite(float(loss.item()))


def test_fully_supervised_batch_matches_plain_cross_entropy_shape():
    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 1]])
    loss = model(tokens, targets=targets)
    assert loss.ndim == 0  # scalar mean loss
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0
