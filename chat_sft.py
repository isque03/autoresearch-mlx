"""Supervised fine-tuning: warm-starts a pretrain checkpoint (train.py's
GPT/AdamW, loaded via checkpoint_manager) on the SmolTalk+MMLU+GSM8K mixture,
same datasets nanochat's own chat_sft.py uses.

Deliberate fixes vs. nanochat's chat_sft.py, both landed here BECAUSE we
root-caused nanochat's own real SFT NaN crash today (stale pretrain-momentum
+ zero LR warmup -> exploding first step):
  1. WARMUP_RATIO defaults to a real nonzero value (0.05), not nanochat's
     default of 0.0.
  2. Optimizer starts with FRESH state (see checkpoint_manager.py's own
     docstring) instead of inheriting pretrain's momentum/variance buffers --
     removes the stale-momentum half of the failure mode entirely, rather
     than just papering over it with a warmup that has to be tuned exactly
     right.

Usage:
    python chat_sft.py <pretrain_checkpoint_dir>
"""
import sys
import time

import mlx.core as mx
from mlx.utils import tree_map

from chat_tokenizer import ChatTokenizer
from checkpoint_manager import build_model_from_checkpoint, save_checkpoint
from prepare import MAX_SEQ_LEN
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from train import GPT, GPTConfig, AdamW

# ---------------------------------------------------------------------------
# Hyperparameters (edit directly, matching this repo's existing style)
# ---------------------------------------------------------------------------
PRETRAIN_CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_extended"
OUTPUT_DIR = "chatsft_checkpoints"

DEVICE_BATCH_SIZE = 8
TOTAL_BATCH_SIZE = 8 * MAX_SEQ_LEN
TIME_BUDGET = 600  # 10 minutes; separate, smaller budget than pretrain's 300s x this repo's own convention of a fixed wall-clock target rather than a step count
MMLU_EPOCHS = 3
GSM8K_EPOCHS = 4

WARMUP_RATIO = 0.05  # nonzero: the actual fix for the NaN bug we found in nanochat's own chat_sft.py
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
INIT_LR_FRAC = 0.5  # start SFT below pretrain's peak LR even after warmup, since fresh optimizer state has no momentum built up yet

EVAL_EVERY = 100
SAMPLE_EVERY = 100
SAMPLE_PROMPTS = (
    "What is the capital of France?",
    "What is 12 + 27?",
    "Explain what a rainbow is in one sentence.",
)


def get_lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return (progress + 1e-8) / WARMUP_RATIO
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    decay = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
    return (1 - decay) * 1.0 + decay * FINAL_LR_FRAC


def render_and_pack(tokenizer: ChatTokenizer, dataset, batch_size: int, seq_len: int):
    """BOS-aligned best-fit packing, same policy as nanochat's SFT dataloader:
    pick the largest still-rendered conversation that fits the remaining row
    space; if none fits, pad the remainder with BOS tokens (mask=0) rather
    than cropping mid-conversation (cropping would corrupt the mask, e.g. by
    cutting off an assistant_end token the model is supposed to learn to
    emit)."""
    row_capacity = seq_len + 1
    bos = tokenizer.get_bos_token_id()
    buffer: list[tuple[list[int], list[int]]] = []
    cursor = 0
    epoch = 1

    def refill():
        nonlocal cursor, epoch
        while len(buffer) < 64:
            if cursor >= len(dataset):
                cursor = 0
                epoch += 1
            ids, mask = tokenizer.render_conversation(dataset[cursor])
            buffer.append((ids, mask))
            cursor += 1

    while True:
        all_ids, all_mask = [], []
        for _ in range(batch_size):
            row_ids, row_mask, pos = [], [], 0
            while pos < row_capacity:
                refill()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, (ids, _) in enumerate(buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx, best_len = i, len(ids)
                if best_idx >= 0:
                    ids, mask = buffer.pop(best_idx)
                    row_ids.extend(ids)
                    row_mask.extend(mask)
                    pos += len(ids)
                else:
                    pad_len = remaining
                    row_ids.extend([bos] * pad_len)
                    row_mask.extend([0] * pad_len)
                    pos += pad_len
            all_ids.append(row_ids[:row_capacity])
            all_mask.append(row_mask[:row_capacity])

        ids_array = mx.array(all_ids, dtype=mx.int32)
        mask_array = mx.array(all_mask, dtype=mx.int8)
        inputs = ids_array[:, :-1]
        targets = ids_array[:, 1:]
        target_mask = mask_array[:, 1:]
        targets = mx.where(target_mask == 0, mx.array(-1, dtype=mx.int32), targets)
        yield inputs, targets, epoch


def sample_and_print(model, tokenizer, engine_cls, max_new_tokens=32):
    from engine import Engine

    terminal_ids = (tokenizer.special_id("<|assistant_end|>"), tokenizer.get_bos_token_id())
    engine = engine_cls(model, tokenizer, terminal_ids)
    for prompt in SAMPLE_PROMPTS:
        conversation = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]}
        prompt_ids = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(prompt_ids, num_samples=1, max_tokens=max_new_tokens, temperature=0.0)
        completion_ids = results[0][len(prompt_ids):]
        print(f"sample: {prompt!r} -> {tokenizer.decode(completion_ids)!r}", flush=True)


def main():
    tokenizer = ChatTokenizer.from_directory()
    model, pretrain_meta = build_model_from_checkpoint(PRETRAIN_CHECKPOINT_DIR, GPT, GPTConfig)
    print(f"Loaded pretrain checkpoint from {PRETRAIN_CHECKPOINT_DIR}, val_bpb={pretrain_meta.get('val_bpb')}", flush=True)

    # Grow the embedding/head tables if the chat tokenizer's vocab is larger
    # than the pretrain tokenizer's (it is: chat adds 9 special tokens on
    # top of prepare.py's own reserved slots) -- new rows fresh-initialized,
    # existing rows carried over unchanged so the pretrained meaning of every
    # shared token id is preserved.
    old_vocab_size = model.config.vocab_size
    new_vocab_size = tokenizer.get_vocab_size()
    if new_vocab_size > old_vocab_size:
        pad = new_vocab_size - old_vocab_size
        scale = 3**0.5 * model.config.n_embd**-0.5 * 0.7
        model.wte.weight = mx.concatenate(
            [model.wte.weight, (mx.random.uniform(-scale, scale, (pad, model.config.n_embd))).astype(model.wte.weight.dtype)], axis=0
        )
        model.lm_head.weight = mx.concatenate(
            [model.lm_head.weight, (mx.random.normal((pad, model.config.n_embd)) * 0.001).astype(model.lm_head.weight.dtype)], axis=0
        )
        model.config.vocab_size = new_vocab_size
        print(f"Grew vocab {old_vocab_size} -> {new_vocab_size} for chat special tokens", flush=True)

    train_tasks = [
        SmolTalk(split="train"),
        *[MMLU(split="auxiliary_train") for _ in range(MMLU_EPOCHS)],
        *[GSM8K(subset="main", split="train") for _ in range(GSM8K_EPOCHS)],
    ]
    dataset = TaskMixture(train_tasks)
    print(f"SFT data mixture: {len(dataset)} conversations", flush=True)

    user_config = pretrain_meta.get("user_config")
    if not user_config:
        raise ValueError(
            "pretrain checkpoint metadata is missing user_config (UNEMBEDDING_LR/"
            "EMBEDDING_LR/MATRIX_LR); SFT must inherit these from the pretrain run, "
            "not from today's train.py constants, which may have since changed."
        )
    optimizer = AdamW(
        model,
        unembedding_lr=user_config["UNEMBEDDING_LR"],
        embedding_lr=user_config["EMBEDDING_LR"],
        matrix_lr=user_config["MATRIX_LR"],
        weight_decay=0.0,  # nanochat also zeroes weight decay for SFT
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    )
    for path in optimizer.initial_lrs:
        optimizer.initial_lrs[path] *= INIT_LR_FRAC
        optimizer.param_config[path]["lr"] = optimizer.initial_lrs[path]

    import mlx.nn as nn
    loss_and_grad = nn.value_and_grad(model, lambda model, inputs, targets: model(inputs, targets=targets))

    loader = render_and_pack(tokenizer, dataset, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)
    tokens_per_step = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_step)

    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9
    min_val_loss = float("inf")

    while total_training_time < TIME_BUDGET:
        t0 = time.time()
        accum_grads = None
        loss_value = None
        for _ in range(grad_accum_steps):
            x, y, _ = next(loader)
            loss, grads = loss_and_grad(model, x, y)
            mx.eval(loss, grads)
            loss_value = loss
            accum_grads = grads if accum_grads is None else tree_map(lambda a, b: a + b, accum_grads, grads)
        if grad_accum_steps > 1:
            accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        optimizer.set_lr_multiplier(lrm)
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters())

        loss_f = float(loss_value.item())
        if not (loss_f == loss_f) or loss_f > 100:  # NaN check (NaN != NaN) alongside the old blow-up threshold
            print(f"FAIL: non-finite or exploded loss at step {step}: {loss_f}", flush=True)
            raise SystemExit(1)

        dt = time.time() - t0
        total_training_time += dt
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        print(
            f"step {step:05d} ({100 * progress:.1f}%) | loss: {debiased_loss:.6f} | lrm: {lrm:.3f} | "
            f"dt: {dt * 1000:.0f}ms | remaining: {max(0.0, TIME_BUDGET - total_training_time):.0f}s",
            flush=True,
        )

        if EVAL_EVERY > 0 and step > 0 and step % EVAL_EVERY == 0:
            print(f"Step {step:05d} | train_loss (smoothed): {debiased_loss:.4f}", flush=True)
        if SAMPLE_EVERY > 0 and step > 0 and step % SAMPLE_EVERY == 0:
            from engine import Engine
            sample_and_print(model, tokenizer, Engine)

        step += 1

    metadata = {
        "step": step,
        "train_loss": debiased_loss,
        "model_config": vars(model.config),
        "user_config": {"warmup_ratio": WARMUP_RATIO, "warmdown_ratio": WARMDOWN_RATIO, "init_lr_frac": INIT_LR_FRAC},
        "pretrain_checkpoint": PRETRAIN_CHECKPOINT_DIR,
    }
    save_checkpoint(OUTPUT_DIR, model, metadata)
    print(f"SFT complete. Checkpoint saved to {OUTPUT_DIR}", flush=True)
    from engine import Engine
    sample_and_print(model, tokenizer, Engine, max_new_tokens=48)


if __name__ == "__main__":
    main()
