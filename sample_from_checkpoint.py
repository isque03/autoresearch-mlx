"""Load a saved checkpoint and sample from it. Model classes copied verbatim
from train_extended.py (NOT imported -- that file is a top-level training
script with no __main__ guard, importing it would kick off training)."""
import math
import sys
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from prepare import MAX_SEQ_LEN, Tokenizer


def set_path_value(model, path, value):
    """Same path-walking logic train_extended.py's AdamW optimizer uses to
    write updated params back -- needed because self.blocks is a plain list
    and self.value_embeds a plain dict, not MLX's ModuleList/ModuleDict, so
    the generic nn.Module.update() tree-matching doesn't handle them."""
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

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_extended/c6bcca6_1.1981.safetensors"
MAX_NEW_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

DEPTH, ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN = 4, 64, 128, "SSSL"


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    is_sink = indices[None, :] == 0
    blocked = (causal | too_far) & ~is_sink
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 1.5 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = norm(self.rope(q))
        k = norm(self.rope(k))
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = x * mx.sigmoid(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config.n_layer,), dtype=mx.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }
        self._mask_cache = {}

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 8
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        dtype = self.wte.weight.dtype
        for window_size in set(self.window_sizes):
            key = (seq_len, window_size, dtype)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len, dtype=dtype)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size, dtype=dtype)
        return [self._mask_cache[(seq_len, window_size, dtype)] for window_size in self.window_sizes]

    def __call__(self, idx):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i].astype(x.dtype) * x + self.x0_lambdas[i].astype(x.dtype) * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)
        return self.lm_head(x).astype(mx.float32)


tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
)
model = GPT(config)
weights = mx.load(CHECKPOINT)
for path, value in weights.items():
    set_path_value(model, path, value)
mx.eval(model.parameters())
print(f"loaded {CHECKPOINT}", flush=True)

prompts = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
for prompt in prompts:
    ids = list(tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id()))
    for _ in range(MAX_NEW_TOKENS):
        logits = model(mx.array([ids]))
        next_id = int(mx.argmax(logits[0, -1]).item())
        ids.append(next_id)
    print(f"\n=== {prompt!r} ===\n{tokenizer.decode(ids)!r}", flush=True)
