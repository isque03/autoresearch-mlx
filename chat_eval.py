"""ChatCORE: evaluates an SFT/chat model on the tasks this repo has data
classes for (MMLU: categorical, GSM8K: generative) -- nanochat's own
chat_eval.py additionally covers ARC-Easy/ARC-Challenge/HumanEval, which
this repo doesn't have task modules for (out of scope per the plan's SFT
data decision: SmolTalk+MMLU+GSM8K only). Same centering-against-
random-baseline algebra as core_eval.py's CORE metric.
"""
from __future__ import annotations

import mlx.core as mx

from engine import Engine

BASELINE_ACCURACIES = {"MMLU": 0.25, "GSM8K": 0.0}


def run_categorical_eval(task, tokenizer, model, max_problems: int | None = None) -> float:
    """Batched forward pass; answer is whichever answer-letter token has the
    highest logit at the answer position (not free-form generation)."""
    n = len(task) if max_problems is None else min(max_problems, len(task))
    correct = 0
    for i in range(n):
        problem = task[i]
        prompt_ids = tokenizer.render_for_completion(problem)
        letters = problem["letters"]
        letter_ids = [tokenizer.encode(letter)[0] for letter in letters]
        input_ids = mx.array([prompt_ids])
        logits = model(input_ids)
        answer_logits = logits[0, -1, :]
        letter_logits = mx.array([float(answer_logits[lid].item()) for lid in letter_ids])
        predicted = letters[int(mx.argmax(letter_logits).item())]
        if task.evaluate(problem, predicted):
            correct += 1
    return correct / n if n > 0 else 0.0


def run_generative_eval(task, tokenizer, model, engine: Engine, max_new_tokens: int = 256, num_samples: int = 1, max_problems: int | None = None) -> float:
    """pass@num_samples: a problem counts as correct if ANY of num_samples
    greedy/sampled completions is correct."""
    n = len(task) if max_problems is None else min(max_problems, len(task))
    correct = 0
    for i in range(n):
        problem = task[i]
        prompt_ids = tokenizer.render_for_completion(problem)
        results, _ = engine.generate_batch(prompt_ids, num_samples=num_samples, max_tokens=max_new_tokens, temperature=0.0)
        outcomes = [task.evaluate(problem, tokenizer.decode(result[len(prompt_ids):])) for result in results]
        if any(outcomes):
            correct += 1
    return correct / n if n > 0 else 0.0


def run_chat_eval(task, tokenizer, model, engine: Engine, **kwargs) -> float:
    if task.eval_type == "categorical":
        return run_categorical_eval(task, tokenizer, model, max_problems=kwargs.get("max_problems"))
    elif task.eval_type == "generative":
        return run_generative_eval(
            task, tokenizer, model, engine,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            num_samples=kwargs.get("num_samples", 1),
            max_problems=kwargs.get("max_problems"),
        )
    raise ValueError(f"unsupported eval_type for chat eval: {task.eval_type}")


def evaluate_chatcore(tasks_by_label: dict, tokenizer, model, engine: Engine, **kwargs) -> dict:
    """tasks_by_label: {"MMLU": mmlu_task_instance, "GSM8K": gsm8k_task_instance}."""
    results, centered = {}, {}
    for label, task in tasks_by_label.items():
        accuracy = run_chat_eval(task, tokenizer, model, engine, **kwargs)
        baseline = BASELINE_ACCURACIES[label]
        results[label] = accuracy
        centered[label] = (accuracy - baseline) / (1.0 - baseline)
        print(f"ChatCORE {label}: accuracy={accuracy:.4f} centered={centered[label]:.4f}", flush=True)
    chatcore = sum(centered.values()) / len(centered)
    return {"results": results, "centered_results": centered, "chatcore": chatcore}
