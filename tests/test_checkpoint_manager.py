"""Checkpoint round-trip tests. Uses train.py's real GPT/GPTConfig (now
safely importable since train.py got a __main__ guard) with a tiny config
so tests stay fast."""
import shutil
import tempfile

import mlx.core as mx
import pytest

from checkpoint_manager import (
    build_model_from_checkpoint, get_flat_params, load_metadata, save_checkpoint,
)
from train import GPT, GPTConfig


def tiny_config(**overrides):
    # n_embd must be >= 32: CausalSelfAttention hardcodes ve_gate_channels=32
    # regardless of n_embd, a real constraint of train.py's current model, not
    # something to work around here.
    defaults = dict(sequence_len=64, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="SL")
    defaults.update(overrides)
    return GPTConfig(**defaults)


@pytest.fixture
def tmp_checkpoint_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_round_trip_is_bit_exact(tmp_checkpoint_dir):
    config = tiny_config()
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())
    original = get_flat_params(model)

    metadata = {"step": 100, "val_bpb": 1.23, "model_config": vars(config)}
    save_checkpoint(tmp_checkpoint_dir, model, metadata)

    loaded_model, loaded_metadata = build_model_from_checkpoint(tmp_checkpoint_dir, GPT, GPTConfig)
    reloaded = get_flat_params(loaded_model)

    assert set(original.keys()) == set(reloaded.keys())
    for path, value in original.items():
        assert mx.array_equal(value, reloaded[path]), f"mismatch at {path}"
    assert loaded_metadata["step"] == 100
    assert loaded_metadata["val_bpb"] == 1.23


def test_loaded_model_produces_identical_forward_pass(tmp_checkpoint_dir):
    config = tiny_config()
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())

    save_checkpoint(tmp_checkpoint_dir, model, {"model_config": vars(config)})
    loaded_model, _ = build_model_from_checkpoint(tmp_checkpoint_dir, GPT, GPTConfig)

    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    original_logits = model(tokens)
    reloaded_logits = loaded_model(tokens)
    assert mx.array_equal(original_logits, reloaded_logits)


def test_metadata_survives_round_trip_with_arbitrary_fields(tmp_checkpoint_dir):
    config = tiny_config()
    model = GPT(config)
    model.init_weights()
    metadata = {
        "step": 5000,
        "val_bpb": 1.234567,
        "model_config": vars(config),
        "user_config": {"embedding_lr": 0.7, "matrix_lr": 0.05, "warmup_ratio": 0.05},
    }
    save_checkpoint(tmp_checkpoint_dir, model, metadata)
    loaded = load_metadata(tmp_checkpoint_dir)
    assert loaded == metadata
