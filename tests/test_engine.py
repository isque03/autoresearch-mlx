"""The highest-value test in the suite: verifies the KV-cache engine
produces EXACTLY the same tokens as naive full-recompute generation (no
cache at all, just re-running the model's own real forward pass on the
growing sequence each step). A subtly wrong cache would otherwise produce
plausible-looking but silently incorrect output -- this is a token-for-token
equivalence check, not an approximate one."""
import mlx.core as mx

from engine import Engine, forward_with_cache, KVCache, sample_next_token
from train import GPT, GPTConfig


def tiny_model(window_pattern="SL"):
    # depth=4 with window_pattern "SL" exercises both the full-context layer
    # and the sliding-window+sink layer within one small model.
    config = GPTConfig(sequence_len=32, vocab_size=48, n_layer=4, n_head=2, n_kv_head=2, n_embd=32, window_pattern=window_pattern)
    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())
    return model


def naive_generate(model, prompt_ids, max_new_tokens):
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model(mx.array([ids]))
        next_id = int(mx.argmax(logits[0, -1]).item())
        ids.append(next_id)
    return ids


def test_kv_cache_matches_naive_generation_greedy():
    model = tiny_model()
    prompt = [1, 2, 3, 4, 5]
    naive_ids = naive_generate(model, prompt, max_new_tokens=10)

    engine = Engine(model, tokenizer=None, terminal_token_ids=())
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=10, temperature=0.0)

    assert results[0] == naive_ids


def test_kv_cache_matches_naive_generation_with_full_window_only():
    model = tiny_model(window_pattern="L")  # every layer full-context, no sliding window/sink path at all
    prompt = [3, 6, 9, 12]
    naive_ids = naive_generate(model, prompt, max_new_tokens=8)

    engine = Engine(model, tokenizer=None, terminal_token_ids=())
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=8, temperature=0.0)

    assert results[0] == naive_ids


def test_kv_cache_matches_naive_generation_longer_than_window():
    # window_size for "S" = sequence_len // 8 = 4 here; generate well past
    # that so the sink+truncation path is actually exercised many times over.
    model = tiny_model(window_pattern="SL")
    prompt = [1, 2]
    naive_ids = naive_generate(model, prompt, max_new_tokens=20)

    engine = Engine(model, tokenizer=None, terminal_token_ids=())
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=20, temperature=0.0)

    assert results[0] == naive_ids


def test_terminal_token_stops_generation_early():
    model = tiny_model()
    prompt = [1, 2, 3]
    naive_ids = naive_generate(model, prompt, max_new_tokens=1)
    terminal_token = naive_ids[-1]  # force this exact token to be "terminal"

    engine = Engine(model, tokenizer=None, terminal_token_ids=(terminal_token,))
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=10, temperature=0.0)

    assert terminal_token not in results[0]  # excluded from output, per generate_batch contract
    assert len(results[0]) == len(prompt)  # stopped immediately, no tokens appended after prompt


def test_multi_sample_batch_shapes_are_consistent():
    model = tiny_model()
    prompt = [1, 2, 3]
    engine = Engine(model, tokenizer=None, terminal_token_ids=())
    results, masks = engine.generate_batch(prompt, num_samples=3, max_tokens=5, temperature=0.0)
    assert len(results) == 3
    for result, mask in zip(results, masks):
        assert len(result) == len(prompt) + 5
        assert len(mask) == len(result)


def test_sample_next_token_top_k_one_does_not_crash_and_matches_argmax():
    # Regression test: top_k=1's threshold slice was `[-1:0]` (empty), which
    # broadcast-crashed instead of ever being exercised -- no prior test used
    # temperature>0 or top_k at all.
    logits = mx.array([[1.0, 5.0, 2.0, 0.5, 3.0]])
    token = sample_next_token(logits, temperature=1.0, top_k=1)
    assert int(token.item()) == int(mx.argmax(logits, axis=-1).item())


def test_sample_next_token_top_k_restricts_to_k_highest_logits():
    logits = mx.array([[1.0, 5.0, 2.0, 0.5, 3.0]])
    top3_ids = {1, 4, 2}  # the three highest-logit indices
    for _ in range(20):
        token = int(sample_next_token(logits, temperature=1.0, top_k=3).item())
        assert token in top3_ids
