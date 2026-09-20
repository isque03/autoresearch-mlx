"""Checkpoint save/load shared by train.py's own top-5 keeper, chat_sft.py,
and any inference/eval script that needs to load a trained model.

Deliberate simplification vs. nanochat's checkpoint manager: we save/restore
MODEL WEIGHTS + METADATA only, not optimizer momentum state. nanochat's own
SFT warm-starts the optimizer's momentum/variance buffers from the end of
pretraining, which is exactly the mechanism we root-caused as the real cause
of its SFT NaN crash today (stale momentum, calibrated under a near-zero
end-of-pretrain LR, combined with a fresh full-size LR step). Starting SFT
with fresh optimizer state sidesteps that whole failure class rather than
requiring a delicately-tuned warmup to compensate for it -- simpler and
safer, at the cost of a slightly slower first few dozen SFT steps while
momentum re-accumulates from scratch. If that cost turns out to matter,
optimizer-state persistence can be added later as its own, separately-tested
feature; this format's `metadata.json` already reserves room for it.
"""
from __future__ import annotations

import json
import os

import mlx.core as mx


def set_path_value(model, path: str, value) -> None:
    """Write a value into `model` at a dotted/indexed path like
    'blocks.0.attn.c_q.weight', walking plain Python lists and dicts as well
    as regular attributes. Needed because MLX's generic nn.Module.update()
    tree-matching does not handle a model whose submodules are stored in a
    plain list (self.blocks) or plain dict (self.value_embeds) rather than
    MLX's ModuleList/ModuleDict -- confirmed by direct reproduction: it
    raises KeyError instead of silently doing the wrong thing, but it also
    doesn't work at all, which is what motivated writing this in the first
    place (see sample_from_checkpoint.py's history)."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def get_flat_params(model) -> dict[str, mx.array]:
    """Inverse of set_path_value: flatten a model's parameters to
    {dotted_path: array}, walking the same plain list/dict structures.
    Equivalent to mlx.utils.tree_flatten(model.parameters()) as a dict,
    provided directly here so save/load don't depend on tree_flatten's
    path-naming matching set_path_value's expectations by coincidence."""
    from mlx.utils import tree_flatten
    return dict(tree_flatten(model.parameters()))


def save_checkpoint(checkpoint_dir: str, model, metadata: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(checkpoint_dir, "model.safetensors"), get_flat_params(model))
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_metadata(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "metadata.json")) as handle:
        return json.load(handle)


def load_weights_into(model, checkpoint_dir: str) -> None:
    """Load model.safetensors from checkpoint_dir into an already-constructed
    model instance (caller is responsible for building a model whose config
    matches metadata.json's "model_config", typically via
    build_model_from_checkpoint below)."""
    weights = mx.load(os.path.join(checkpoint_dir, "model.safetensors"))
    for path, value in weights.items():
        set_path_value(model, path, value)
    mx.eval(model.parameters())


def build_model_from_checkpoint(checkpoint_dir: str, gpt_cls, gpt_config_cls):
    """Construct a fresh model from metadata.json's saved "model_config" and
    load its weights. gpt_cls/gpt_config_cls are passed in rather than
    imported directly to avoid a hard import-time dependency on train.py's
    specific class shape from this module (train.py's architecture is what
    the autoresearch loop mutates; this file should not need to change when
    it does)."""
    metadata = load_metadata(checkpoint_dir)
    config = gpt_config_cls(**metadata["model_config"])
    model = gpt_cls(config)
    load_weights_into(model, checkpoint_dir)
    return model, metadata
