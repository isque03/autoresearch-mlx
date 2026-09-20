"""KV-cache inference engine. Mirrors nanochat's Engine behavioral contract
(prefill once, decode incrementally, per-row early stopping, forced-token
injection for tool-call output) without depending on train.py's internals
beyond its public module attributes -- this file does NOT modify train.py
(the autoresearch loop's live, actively-mutated file). Instead it duplicates
the small amount of per-layer attention math (projections, RoPE, value-embed
gating, sliding-window+sink key selection) needed to write into a cache,
mirroring train.py's CausalSelfAttention/Block/GPT.__call__ exactly. The
test suite's cached-vs-naive equivalence check exists specifically to catch
any drift between this copy and train.py's real forward pass.

Design choices vs. nanochat's FA3-shaped cache: no framework-specific cache
layout is needed here (that layout exists purely to match FA3's in-place
update API) -- a plain per-layer (B, T, H, D) array pair grown via
mx.concatenate is simple and fast enough at these model sizes on MLX's
lazy-eval engine.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


@dataclass
class KVCache:
    n_layers: int
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)
    length: int = 0

    def __post_init__(self):
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers

    def append(self, layer_idx: int, new_k: mx.array, new_v: mx.array) -> None:
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = new_k, new_v
        else:
            self.k[layer_idx] = mx.concatenate([self.k[layer_idx], new_k], axis=1)
            self.v[layer_idx] = mx.concatenate([self.v[layer_idx], new_v], axis=1)

    def select_for_window(self, layer_idx: int, window_size: int, seq_len: int):
        """Returns (k, v) restricted to what this layer's attention window
        can see: the full cache if window_size >= seq_len (a long/full-context
        layer), otherwise the attention-sink token (position 0) concatenated
        with the most recent (window_size - 1) positions -- matching
        train.py's create_sliding_window_mask sink exception exactly."""
        full_k, full_v = self.k[layer_idx], self.v[layer_idx]
        total = full_k.shape[1]
        if window_size >= seq_len or total <= window_size:
            return full_k, full_v
        recent = window_size - 1
        sink_k, sink_v = full_k[:, :1], full_v[:, :1]
        recent_k, recent_v = full_k[:, -recent:], full_v[:, -recent:]
        return mx.concatenate([sink_k, recent_k], axis=1), mx.concatenate([sink_v, recent_v], axis=1)


def _attention_with_cache(attn, x, ve, cache: KVCache, layer_idx: int, window_size: int, seq_len: int):
    """Mirrors train.py::CausalSelfAttention.__call__ exactly, but writes
    (post-RoPE, post-norm) k/v into `cache` instead of building a static
    mask over the whole sequence -- x is only the NEW chunk (the full prompt
    during prefill, or exactly one token during decode)."""
    batch_size, seq_chunk, _ = x.shape
    q = attn.c_q(x).reshape(batch_size, seq_chunk, attn.n_head, attn.head_dim)
    k = attn.c_k(x).reshape(batch_size, seq_chunk, attn.n_kv_head, attn.head_dim)
    v = attn.c_v(x).reshape(batch_size, seq_chunk, attn.n_kv_head, attn.head_dim)

    if ve is not None and attn.ve_gate is not None:
        ve = ve.reshape(batch_size, seq_chunk, attn.n_kv_head, attn.head_dim)
        gate = 1.5 * mx.sigmoid(attn.ve_gate(x[..., : attn.ve_gate_channels]))
        v = v + mx.expand_dims(gate, axis=-1) * ve

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    offset = cache.length
    q = norm(attn.rope(q, offset=offset))
    k = norm(attn.rope(k, offset=offset))

    cache.append(layer_idx, k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))  # store as (B, T, H, D)
    scale = 1.0 / math.sqrt(attn.head_dim)
    is_prefill = seq_chunk > 1

    if is_prefill:
        # Different query positions within the chunk need DIFFERENT windows
        # (a proper banded mask), so the decode-path's fixed sink+recent
        # truncation is wrong here -- use the full (freshly-written, so
        # exactly this chunk's own) k/v with an explicit per-position
        # causal+window+sink mask, matching train.py's
        # create_sliding_window_mask exactly. This relies on the cache being
        # empty before this call (offset == 0) -- prefill into a non-empty
        # cache would need to mask against prior history too, which this
        # branch does not do.
        assert offset == 0, "prefill path assumes an empty KVCache; incremental multi-turn prefill is not implemented"
        full_k = cache.k[layer_idx].transpose(0, 2, 1, 3)
        full_v = cache.v[layer_idx].transpose(0, 2, 1, 3)
        indices = mx.arange(seq_chunk)
        causal = indices[None, :] > indices[:, None]
        if window_size >= seq_len:
            blocked = causal
        else:
            too_far = (indices[:, None] - indices[None, :]) >= window_size
            is_sink = indices[None, :] == 0
            blocked = (causal | too_far) & ~is_sink
        attn_mask = mx.where(blocked, mx.array(float("-inf"), dtype=x.dtype), mx.array(0.0, dtype=x.dtype))
    else:
        # Exactly one new query position (the common decode case): select
        # only the keys its window can see -- always the sink (position 0)
        # plus the most recent (window_size - 1) cached positions. All
        # selected positions are already in the past (or self), so no
        # additional masking is needed.
        full_k, full_v = cache.select_for_window(layer_idx, window_size, seq_len)
        full_k = full_k.transpose(0, 2, 1, 3)
        full_v = full_v.transpose(0, 2, 1, 3)
        attn_mask = None
    y = mx.fast.scaled_dot_product_attention(q, full_k, full_v, scale=scale, mask=attn_mask)
    y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_chunk, -1)
    return attn.c_proj(y)


def forward_with_cache(model, idx: mx.array, cache: KVCache) -> mx.array:
    """Runs `idx` (shape (B, T)) through `model` using/updating `cache`,
    returning logits at every position of this chunk. Call once with the
    full prompt (prefill, T = prompt length), then repeatedly with T=1 for
    each decode step."""
    x = model.wte(idx)
    x = norm(x)
    x0 = x
    seq_len = model.config.sequence_len
    for i, block in enumerate(model.blocks):
        x = model.resid_lambdas[i].astype(x.dtype) * x + model.x0_lambdas[i].astype(x.dtype) * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        window_size = model.window_sizes[i]
        attn_out = _attention_with_cache(block.attn, norm(x), ve, cache, i, window_size, seq_len)
        x = x + attn_out
        x = x + block.mlp(norm(x))
    x = norm(x)
    cache.length += idx.shape[1]
    return model.lm_head(x).astype(mx.float32)


def sample_next_token(logits: mx.array, temperature: float = 0.0, top_k: int | None = None) -> mx.array:
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    scaled = logits / temperature
    if top_k is not None:
        threshold = mx.sort(scaled, axis=-1)[..., -top_k][..., None]
        scaled = mx.where(scaled < threshold, mx.array(float("-inf"), dtype=scaled.dtype), scaled)
    probs = mx.softmax(scaled, axis=-1)
    return mx.random.categorical(mx.log(probs))


@dataclass
class RowState:
    tokens: list
    forced: deque = field(default_factory=deque)
    completed: bool = False


class Engine:
    def __init__(self, model, tokenizer, terminal_token_ids: tuple[int, ...]):
        self.model = model
        self.tokenizer = tokenizer
        self.terminal_token_ids = set(terminal_token_ids)

    def generate(self, prompt_ids: list[int], num_samples: int = 1, max_tokens: int = 64, temperature: float = 0.0, top_k: int | None = None):
        """Streaming generator: yields (token_column, mask_column), each of
        length num_samples, one step at a time."""
        cache = KVCache(n_layers=len(self.model.blocks))
        prompt = mx.array([prompt_ids])
        logits = forward_with_cache(self.model, prompt, cache)
        last_logits = logits[:, -1, :]  # (1, vocab)

        if num_samples > 1:
            last_logits = mx.broadcast_to(last_logits, (num_samples, last_logits.shape[-1]))
            for layer in range(len(cache.k)):
                cache.k[layer] = mx.broadcast_to(cache.k[layer], (num_samples,) + cache.k[layer].shape[1:])
                cache.v[layer] = mx.broadcast_to(cache.v[layer], (num_samples,) + cache.v[layer].shape[1:])

        rows = [RowState(tokens=list(prompt_ids)) for _ in range(num_samples)]
        for _ in range(max_tokens):
            # token_column is what callers see (the real sampled/forced token,
            # even if it turns out to be terminal -- generate_batch needs the
            # real value to recognize and exclude it). feed_column is what
            # actually goes into the next forward pass: a harmless filler for
            # any row that just completed or was already done, since we must
            # not keep decoding a finished row's content forward.
            token_column, mask_column, feed_column = [], [], []
            for i, row in enumerate(rows):
                if row.completed:
                    token_column.append(0)
                    mask_column.append(0)
                    feed_column.append(0)
                    continue
                if row.forced:
                    token = row.forced.popleft()
                    mask_column.append(0)
                else:
                    token = int(sample_next_token(last_logits[i:i + 1], temperature, top_k).item())
                    mask_column.append(1)
                token_column.append(token)
                if token in self.terminal_token_ids:
                    row.completed = True
                    feed_column.append(0)
                else:
                    row.tokens.append(token)
                    feed_column.append(token)
            yield token_column, mask_column
            if all(row.completed for row in rows):
                break
            next_input = mx.array(feed_column).reshape(num_samples, 1)
            logits = forward_with_cache(self.model, next_input, cache)
            last_logits = logits[:, -1, :]

    def generate_batch(self, prompt_ids: list[int], num_samples: int = 1, **kwargs):
        results = [list(prompt_ids) for _ in range(num_samples)]
        masks = [[0] * len(prompt_ids) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, mask_column in self.generate(prompt_ids, num_samples=num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, mask_column)):
                if completed[i]:
                    continue
                if token in self.terminal_token_ids:
                    completed[i] = True
                else:
                    results[i].append(token)
                    masks[i].append(mask)
            if all(completed):
                break
        return results, masks
