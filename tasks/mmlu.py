"""MMLU: 4-choice multiple-choice knowledge questions across 57 subjects.
Mirrors nanochat's tasks/mmlu.py -- categorical eval_type, exact-letter-match
scoring."""
from __future__ import annotations

from tasks.common import HubDataset, Task, render_mc

LETTERS = ("A", "B", "C", "D")


class MMLU(Task):
    def __init__(self, subset: str = "all", split: str = "auxiliary_train", **kwargs):
        assert split in ("auxiliary_train", "validation", "dev", "test")
        self._data = HubDataset("cais/mmlu", subset=subset, split=split)
        super().__init__(**kwargs)

    @property
    def eval_type(self) -> str:
        return "categorical"

    def num_examples(self) -> int:
        return len(self._data)

    def get_example(self, index: int) -> dict:
        row = self._data[index]
        question = row["question"]
        choices = row["choices"]
        answer_index = row["answer"]
        return {
            "messages": [
                {"role": "user", "content": render_mc(question, LETTERS, choices)},
                {"role": "assistant", "content": LETTERS[answer_index]},
            ],
            "letters": LETTERS,
            "subject": row.get("subject", ""),
        }

    def evaluate(self, problem: dict, completion: str) -> bool:
        assert completion in problem["letters"], f"response {completion!r} is not one of {problem['letters']}"
        ground_truth = problem["messages"][-1]["content"]
        return completion == ground_truth
