"""Tests for gsm8k.py/mmlu.py/smoltalk.py's parsing/rendering/scoring logic.
No network access: each Task's HubDataset dependency is monkeypatched with
a tiny in-memory fake exposing the same __len__/__getitem__ interface."""
from tasks.gsm8k import GSM8K, _extract_final_answer, _parse_answer_to_parts
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk, _validate_messages


class FakeHubDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def make_task_with_fake_data(task_cls, fake_rows, **kwargs):
    task = task_cls.__new__(task_cls)
    task._data = FakeHubDataset(fake_rows)
    from tasks.common import Task
    Task.__init__(task, **kwargs)
    return task


# --- GSM8K -------------------------------------------------------------

def test_extract_final_answer_strips_commas():
    assert _extract_final_answer("blah blah\n#### 1,234") == "1234"


def test_extract_final_answer_handles_negative():
    assert _extract_final_answer("#### -42") == "-42"


def test_extract_final_answer_missing_returns_none():
    assert _extract_final_answer("no marker here") is None


def test_parse_answer_splits_calculator_annotations_into_tool_parts():
    raw = "Natalia sold 48 clips.\n<<48/2=24>>\nShe sold 24 more.\n#### 24"
    parts = _parse_answer_to_parts(raw)
    types = [part["type"] for part in parts]
    assert "python" in types and "python_output" in types
    python_part = next(p for p in parts if p["type"] == "python")
    output_part = next(p for p in parts if p["type"] == "python_output")
    assert python_part["text"] == "48/2"
    assert output_part["text"] == "24"


def test_gsm8k_get_example_shape_and_evaluate():
    rows = [{"question": "What is 2+2?", "answer": "Compute it.\n<<2+2=4>>\n#### 4"}]
    task = make_task_with_fake_data(GSM8K, rows)
    example = task[0]
    assert example["messages"][0] == {"role": "user", "content": "What is 2+2?"}
    assert example["ground_truth_answer"] == "4"
    assert task.evaluate(example, "the answer is #### 4") is True
    assert task.evaluate(example, "the answer is #### 5") is False


# --- MMLU ----------------------------------------------------------------

def test_mmlu_get_example_shape():
    rows = [{"question": "Sky color?", "choices": ["red", "blue", "green", "yellow"], "answer": 1, "subject": "science"}]
    task = make_task_with_fake_data(MMLU, rows)
    example = task[0]
    assert example["messages"][-1] == {"role": "assistant", "content": "B"}
    assert example["letters"] == ("A", "B", "C", "D")
    assert "blue=B" in example["messages"][0]["content"]


def test_mmlu_evaluate_exact_match():
    rows = [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": "s"}]
    task = make_task_with_fake_data(MMLU, rows)
    example = task[0]
    assert task.evaluate(example, "A") is True
    assert task.evaluate(example, "B") is False


def test_mmlu_evaluate_rejects_non_letter_response():
    rows = [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": "s"}]
    task = make_task_with_fake_data(MMLU, rows)
    example = task[0]
    try:
        task.evaluate(example, "not a letter")
        assert False, "expected an assertion error"
    except AssertionError:
        pass


# --- SmolTalk --------------------------------------------------------------

def test_smoltalk_passthrough_shape():
    rows = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]
    task = make_task_with_fake_data(SmolTalk, rows)
    example = task[0]
    assert example["messages"] == rows[0]["messages"]


def test_smoltalk_validates_alternation():
    bad = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    try:
        _validate_messages(bad)
        assert False, "expected an assertion error"
    except AssertionError:
        pass


def test_smoltalk_validates_alternation_allows_leading_system():
    good = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    _validate_messages(good)  # should not raise
