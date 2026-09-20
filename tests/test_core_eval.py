"""Tests for core_eval.py's prompt rendering, common-prefix/suffix trimming,
and centering algebra. No network access -- doesn't touch the real
eval_bundle.zip download."""
import mlx.core as mx

from core_eval import (
    batch_sequences_lm, batch_sequences_mc, batch_sequences_schema,
    find_common_length, forward_losses_and_predictions, render_prompts_lm,
    render_prompts_mc, render_prompts_schema,
)
from train import GPT, GPTConfig


class FakeTokenizer:
    """Character-level fake tokenizer: encode(str) -> list[int] (ord of each
    char), so common-prefix/suffix length is easy to reason about by hand."""

    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        if isinstance(text, str):
            ids = [ord(c) for c in text]
        else:
            ids = [[ord(c) for c in t] for t in text]
        if prepend is not None:
            if isinstance(text, str):
                ids = [prepend] + ids
            else:
                ids = [[prepend] + row for row in ids]
        return ids


def test_render_prompts_mc_one_prompt_per_choice():
    item = {"query": "2+2=", "choices": ["3", "4"]}
    prompts = render_prompts_mc(item, " ", fewshot_examples=None)
    assert prompts == ["2+2= 3", "2+2= 4"]


def test_render_prompts_mc_includes_fewshot_prefix():
    item = {"query": "2+2=", "choices": ["4"]}
    fewshot = [{"query": "1+1=", "choices": ["2", "3"], "gold": 0}]
    prompts = render_prompts_mc(item, " ", fewshot_examples=fewshot)
    assert prompts == ["1+1= 2\n\n2+2= 4"]


def test_render_prompts_schema_one_prompt_per_context_option():
    item = {"context_options": ["The cat sat", "The dog sat"], "continuation": " on the mat"}
    prompts = render_prompts_schema(item, "", fewshot_examples=None)
    assert prompts == ["The cat sat on the mat", "The dog sat on the mat"]


def test_render_prompts_lm_returns_without_and_with_continuation():
    item = {"context": "the sky is  ", "continuation": "blue"}
    without, with_ = render_prompts_lm(item, " ", fewshot_examples=None)
    assert with_.endswith("blue")
    assert not without.endswith("blue")
    assert with_.startswith(without) or with_[: len(without)] == without


def test_find_common_length_left_prefix():
    sequences = [[1, 2, 3, 7, 8], [1, 2, 3, 9]]
    assert find_common_length(sequences, direction="left") == 3


def test_find_common_length_right_suffix():
    sequences = [[5, 1, 2, 3], [9, 9, 1, 2, 3]]
    assert find_common_length(sequences, direction="right") == 3


def test_batch_sequences_mc_finds_shared_context_prefix():
    tokenizer = FakeTokenizer()
    prompts = ["2+2= 3", "2+2= 4"]
    tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    # bos + "2+2= " is the shared prefix across both prompts
    shared_prefix_len = 1 + len("2+2= ")
    assert start_idxs == [shared_prefix_len, shared_prefix_len]
    assert end_idxs == [len(tokens[0]), len(tokens[1])]


def test_batch_sequences_schema_finds_shared_continuation_suffix():
    tokenizer = FakeTokenizer()
    prompts = ["The cat sat on the mat", "The dog sat on the mat"]
    tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    suffix = " sat on the mat"  # "cat"/"dog" is where they diverge; everything after is shared
    for start, end, token_row in zip(start_idxs, end_idxs, tokens):
        assert token_row[start:end] == [ord(c) for c in suffix]


def test_batch_sequences_lm_prefix_consistency_assertion_holds():
    tokenizer = FakeTokenizer()
    prompts = ["the sky is", "the sky is blue"]
    tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    assert len(tokens) == 1  # LM task batches to size 1
    assert start_idxs[0] < end_idxs[0]


def test_centering_formula_zero_at_baseline_one_at_perfect():
    def centered(accuracy, baseline_pct):
        return (accuracy - 0.01 * baseline_pct) / (1.0 - 0.01 * baseline_pct)

    assert abs(centered(0.25, 25.0) - 0.0) < 1e-9  # exactly at random baseline -> 0
    assert abs(centered(1.0, 25.0) - 1.0) < 1e-9   # perfect -> 1
    assert centered(0.5, 25.0) > 0  # better than random -> positive


def test_forward_losses_and_predictions_shapes():
    config = GPTConfig(sequence_len=32, vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    losses, predictions = forward_losses_and_predictions(model, input_ids)
    assert losses.shape == (1, 5)
    assert predictions.shape == (1, 5)
