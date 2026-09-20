"""Tests for tasks/common.py's Task/TaskMixture/TaskSequence/render_mc.
No network access needed -- HubDataset (the only network-touching piece) is
exercised separately, gated behind real download, not here."""
from tasks.common import Task, TaskMixture, TaskSequence, render_mc


class FakeTask(Task):
    def __init__(self, tag: str, n: int):
        self.tag = tag
        self._n = n
        super().__init__()

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self._n

    def get_example(self, index):
        return {"tag": self.tag, "index": index}


def test_task_len_and_getitem():
    task = FakeTask("a", 5)
    assert len(task) == 5
    assert task[0] == {"tag": "a", "index": 0}
    assert task[4] == {"tag": "a", "index": 4}


def test_task_getitem_out_of_range_raises():
    task = FakeTask("a", 3)
    try:
        task[3]
        assert False, "expected IndexError"
    except IndexError:
        pass


def test_task_mixture_covers_every_example_from_every_task():
    mixture = TaskMixture([FakeTask("a", 3), FakeTask("b", 2)])
    assert len(mixture) == 5
    seen = {(item["tag"], item["index"]) for item in (mixture[i] for i in range(len(mixture)))}
    expected = {("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1)}
    assert seen == expected


def test_task_mixture_shuffle_is_deterministic():
    mixture_a = TaskMixture([FakeTask("a", 10)])
    mixture_b = TaskMixture([FakeTask("a", 10)])
    order_a = [mixture_a[i]["index"] for i in range(len(mixture_a))]
    order_b = [mixture_b[i]["index"] for i in range(len(mixture_b))]
    assert order_a == order_b


def test_task_mixture_oversampling_via_repeated_task_instances():
    base_task = FakeTask("a", 2)
    mixture = TaskMixture([base_task, base_task, base_task])  # 3 epochs of the same task
    assert len(mixture) == 6


def test_task_sequence_preserves_task_order_not_shuffled():
    sequence = TaskSequence([FakeTask("a", 3), FakeTask("b", 2)])
    assert len(sequence) == 5
    items = [sequence[i] for i in range(len(sequence))]
    assert [item["tag"] for item in items] == ["a", "a", "a", "b", "b"]
    assert [item["index"] for item in items] == [0, 1, 2, 0, 1]


def test_render_mc_letter_follows_choice_with_no_leading_space():
    prompt = render_mc("What color is the sky?", ("A", "B"), ["blue", "green"])
    assert "blue=A" in prompt
    assert "green=B" in prompt
    assert " =A" not in prompt  # no stray space before the letter
