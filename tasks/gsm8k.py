"""GSM8K: grade-school math word problems with calculator-annotated
solutions. Mirrors nanochat's tasks/gsm8k.py -- splits the raw answer's
<<expr=result>> calculator annotations into tool-call parts so the SFT
target teaches the model to invoke the calculator rather than compute
arithmetic purely in its own weights."""
from __future__ import annotations

import re

from tasks.common import HubDataset, Task

CALC_PATTERN = re.compile(r"(<<[^>]+>>)")
FINAL_ANSWER_PATTERN = re.compile(r"#### (-?[0-9.,]+)")


def _parse_answer_to_parts(answer: str) -> list[dict]:
    parts = []
    for chunk in CALC_PATTERN.split(answer):
        if not chunk:
            continue
        if chunk.startswith("<<") and chunk.endswith(">>"):
            inner = chunk[2:-2]  # "expr=result"
            expr, _, result = inner.partition("=")
            parts.append({"type": "python", "text": expr})
            parts.append({"type": "python_output", "text": result})
        else:
            parts.append({"type": "text", "text": chunk})
    return parts


def _extract_final_answer(text: str) -> str | None:
    match = FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


class GSM8K(Task):
    def __init__(self, subset: str = "main", split: str = "train", **kwargs):
        assert subset in ("main", "socratic")
        assert split in ("train", "test")
        self._data = HubDataset("openai/gsm8k", subset=subset, split=split)
        super().__init__(**kwargs)

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return len(self._data)

    def get_example(self, index: int) -> dict:
        row = self._data[index]
        return {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": _parse_answer_to_parts(row["answer"])},
            ],
            "ground_truth_answer": _extract_final_answer(row["answer"]),
        }

    def evaluate(self, problem: dict, completion: str) -> bool:
        predicted = _extract_final_answer(completion)
        return predicted is not None and predicted == problem["ground_truth_answer"]

    def reward(self, problem: dict, completion: str) -> float:
        return 1.0 if self.evaluate(problem, completion) else 0.0
