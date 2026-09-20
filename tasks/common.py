"""Task abstraction shared by gsm8k.py/mmlu.py/smoltalk.py: a Task is a
dataset of conversation dicts (see chat_tokenizer.py's expected schema).
Mirrors nanochat's tasks/common.py (TaskMixture's shuffle, HubDataset's
parquet-download-and-cache shape) so the SFT data mixture behaves the same
way nanochat's does."""
from __future__ import annotations

import math
import os
import random

import pyarrow.parquet as pq
import requests

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TASK_DATA_DIR = os.path.join(CACHE_DIR, "task_data")


class Task:
    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1):
        self.start = start
        self.stop = stop if stop is not None else self.num_examples()
        self.step = step

    @property
    def eval_type(self) -> str:
        raise NotImplementedError  # 'generative' | 'categorical'

    def num_examples(self) -> int:
        raise NotImplementedError

    def get_example(self, index: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return math.ceil((self.stop - self.start) / self.step)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        physical_index = self.start + index * self.step
        return self.get_example(physical_index)

    def evaluate(self, problem: dict, completion: str) -> bool:
        raise NotImplementedError


class TaskMixture:
    """Concatenates multiple Tasks into one, examples interleaved via a
    deterministic shuffle (seed=42) fixed at construction time -- so
    oversampling a task (passing it multiple times) spreads its repeats
    throughout an epoch instead of clustering them at the end."""

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        index_map = []
        for task_idx, task in enumerate(tasks):
            index_map.extend((task_idx, local_idx) for local_idx in range(len(task)))
        random.Random(42).shuffle(index_map)
        self.index_map = index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict:
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence:
    """Like TaskMixture but preserves task order (no shuffling) -- walks
    task 1 fully, then task 2, etc."""

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self.lengths = [len(task) for task in tasks]
        self.cumulative = [0]
        for length in self.lengths:
            self.cumulative.append(self.cumulative[-1] + length)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict:
        for task_idx, length in enumerate(self.lengths):
            if index < self.cumulative[task_idx + 1]:
                return self.tasks[task_idx][index - self.cumulative[task_idx]]
        raise IndexError(index)


class HubDataset:
    """Minimal HF-hub-parquet dataset loader: downloads a dataset repo's
    parquet shard(s) via the public HF API (no `datasets` package
    dependency, matching prepare.py's own from-scratch download style),
    caches under TASK_DATA_DIR, loads via pyarrow."""

    def __init__(self, repo_id: str, subset: str = "default", split: str = "train"):
        self.repo_id = repo_id
        self.subset = subset
        self.split = split
        self._table = self._load()

    def _cache_dir(self) -> str:
        repo_slug = self.repo_id.replace("/", "__")
        return os.path.join(TASK_DATA_DIR, repo_slug, self.subset, self.split)

    def _load(self):
        cache_dir = self._cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, "data.parquet")
        if not os.path.exists(local_path):
            self._download(local_path)
        return pq.read_table(local_path)

    def _download(self, local_path: str) -> None:
        url = (
            f"https://huggingface.co/api/datasets/{self.repo_id}/parquet/"
            f"{self.subset}/{self.split}/0.parquet"
        )
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        os.rename(tmp_path, local_path)

    def shuffle(self, seed: int) -> "HubDataset":
        import numpy as np
        permutation = np.random.default_rng(seed).permutation(len(self))
        shuffled = HubDataset.__new__(HubDataset)
        shuffled.repo_id, shuffled.subset, shuffled.split = self.repo_id, self.subset, self.split
        shuffled._table = self._table.take(permutation)
        return shuffled

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int) -> dict:
        row = self._table.slice(index, 1).to_pylist()[0]
        return row


def render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    """Canonical multiple-choice prompt: letter goes AFTER each choice
    (better binding for small models, per nanochat's own design note), and
    with no leading space before the letter, since the assistant's answer
    token (the bare letter, e.g. "A") must tokenize identically to how it
    appears here."""
    lines = [f"Multiple Choice question: {question}"]
    for letter, choice in zip(letters, choices):
        lines.append(f"- {choice}={letter}")
    lines.append("Respond only with the letter of the correct answer.")
    return "\n".join(lines)
