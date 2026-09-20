"""CORE benchmark (DCLM paper, https://arxiv.org/abs/2406.11794): evaluates
a BASE (pretrained, non-chat) model on ~20 standard few-shot ICL tasks,
scored via continuation log-likelihood and centered against each task's
random-guess baseline so tasks with different numbers of answer choices are
comparable. Ports nanochat's nanochat/core_eval.py + scripts/base_eval.py's
evaluate_core algorithm to MLX. Uses the SAME eval_bundle.zip nanochat
downloads (task list + per-task JSONL + baseline-accuracy CSV) -- that's a
stable, versioned artifact, not something to re-derive.

Prompt rendering uses plain string formatting rather than nanochat's jinja2
templates -- the three templates are simple enough that a templating engine
is an unnecessary dependency for reproducing them exactly.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
import zipfile
from urllib.request import urlretrieve

import mlx.core as mx
import mlx.nn as nn
import yaml

from prepare import CACHE_DIR

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR = os.path.join(CACHE_DIR, "eval_bundle")


def ensure_eval_bundle() -> None:
    if os.path.exists(EVAL_BUNDLE_DIR):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "eval_bundle.zip")
    print(f"Downloading CORE eval bundle from {EVAL_BUNDLE_URL}...", flush=True)
    urlretrieve(EVAL_BUNDLE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(CACHE_DIR)
    print(f"Eval bundle ready at {EVAL_BUNDLE_DIR}", flush=True)


# --- Prompt rendering (plain string formatting, matches nanochat's jinja2 templates exactly) ---

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{example['query']}{continuation_delimiter}{example['choices'][example['gold']]}\n\n"
        for example in fewshot_examples
    )
    return [f"{prefix}{item['query']}{continuation_delimiter}{choice}" for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{example['context_options'][example['gold']]}{continuation_delimiter}{example['continuation']}\n\n"
        for example in fewshot_examples
    )
    return [f"{prefix}{context_option}{continuation_delimiter}{item['continuation']}" for context_option in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{example['context'].strip()}{continuation_delimiter}{example['continuation']}\n\n"
        for example in fewshot_examples
    )
    prompt_without = f"{prefix}{item['context'].strip()}{continuation_delimiter}".strip()
    prompt_with = f"{prefix}{item['context'].strip()}{continuation_delimiter}{item['continuation']}"
    return [prompt_without, prompt_with]


def find_common_length(token_sequences: list[list[int]], direction: str = "left") -> int:
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start = find_common_length(tokens, direction="left")
    return tokens, [answer_start] * len(prompts), [len(t) for t in tokens]


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(t) for t in tokens]
    return tokens, [e - suffix_length for e in end_indices], end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert tokens_without == tokens_with[:start_idx]
    return [tokens_with], [start_idx], [end_idx]


def stack_sequences(tokens: list[list[int]], pad_token_id: int) -> mx.array:
    seq_len = max(len(t) for t in tokens)
    padded = [t + [pad_token_id] * (seq_len - len(t)) for t in tokens]
    return mx.array(padded, dtype=mx.int32)


def forward_losses_and_predictions(model, input_ids: mx.array):
    """Returns (losses, predictions), both (B, T): losses[b, t] is the
    cross-entropy of predicting input_ids[b, t+1] from position t (last
    column is NaN, no target exists there); predictions[b, t] is the
    argmax next-token prediction at position t."""
    batch_size, seq_len = input_ids.shape
    logits = model(input_ids)
    targets = mx.concatenate([input_ids[:, 1:], input_ids[:, :1]], axis=1)  # roll left by 1
    losses_flat = nn.losses.cross_entropy(
        logits.reshape(batch_size * seq_len, -1), targets.reshape(-1), reduction="none"
    )
    losses = losses_flat.reshape(batch_size, seq_len)
    predictions = mx.argmax(logits, axis=-1)
    return losses, predictions


def evaluate_example(idx: int, model, tokenizer, data: list[dict], task_meta: dict) -> bool:
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    item = data[idx]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot_examples = [data[i] for i in rng.sample(available, num_fewshot)]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    max_tokens = getattr(model.config, "sequence_len", None)
    if max_tokens is not None:
        cropped = []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                drop = len(t) - max_tokens
                cropped.append((t[-max_tokens:], s - drop, e - drop))
                assert s - drop >= 0 and e - drop >= 0
            else:
                cropped.append((t, s, e))
        tokens, start_idxs, end_idxs = zip(*cropped)
        tokens, start_idxs, end_idxs = list(tokens), list(start_idxs), list(end_idxs)

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    losses, predictions = forward_losses_and_predictions(model, input_ids)

    if task_type == "language_modeling":
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si - 1:ei - 1]
        actual_tokens = input_ids[0, si:ei]
        return bool(mx.all(predicted_tokens == actual_tokens).item())
    else:
        mean_losses = [
            float(mx.mean(losses[i, si - 1:ei - 1]).item()) for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        predicted_idx = mean_losses.index(min(mean_losses))
        return predicted_idx == item["gold"]


def evaluate_task(model, tokenizer, data: list[dict], task_meta: dict) -> float:
    correct = [evaluate_example(idx, model, tokenizer, data, task_meta) for idx in range(len(data))]
    return sum(correct) / len(correct)


def evaluate_core(model, tokenizer, max_per_task: int = -1) -> dict:
    ensure_eval_bundle()
    with open(os.path.join(EVAL_BUNDLE_DIR, "core.yaml")) as handle:
        config = yaml.safe_load(handle)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(os.path.join(EVAL_BUNDLE_DIR, "eval_meta_data.csv")) as handle:
        for row in csv.DictReader(handle):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results, centered_results = {}, {}
    for task in tasks:
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = os.path.join(EVAL_BUNDLE_DIR, "eval_data", task["dataset_uri"])
        with open(data_path) as handle:
            data = [json.loads(line) for line in handle]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        t0 = time.time()
        accuracy = evaluate_task(model, tokenizer, data, task_meta)
        baseline = random_baselines[label]
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        results[label] = accuracy
        centered_results[label] = centered
        print(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... "
            f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {time.time() - t0:.2f}s",
            flush=True,
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}
