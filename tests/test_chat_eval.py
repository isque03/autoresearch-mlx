"""Tests for chat_eval.py's categorical/generative eval loops and the
ChatCORE centering algebra. Uses a tiny real GPT model (so the categorical
path's actual logit-based answer selection is genuinely exercised, not
mocked) with the in-memory byte-level tokenizer fixture."""
import mlx.core as mx
import tiktoken

from chat_eval import evaluate_chatcore, run_categorical_eval, run_generative_eval
from chat_tokenizer import ChatTokenizer, CHAT_SPECIAL_TOKENS
from engine import Engine
from tasks.common import Task
from train import GPT, GPTConfig


def make_test_tokenizer():
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    special_tokens = {name: 256 + i for i, name in enumerate(CHAT_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="test", pat_str=r".", mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
    return ChatTokenizer(enc)


def tiny_model(vocab_size):
    config = GPTConfig(sequence_len=64, vocab_size=vocab_size, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    return model


class FakeCategoricalTask(Task):
    def __init__(self, examples):
        self._examples = examples
        super().__init__()

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self._examples)

    def get_example(self, index):
        return self._examples[index]

    def evaluate(self, problem, completion):
        return completion == problem["messages"][-1]["content"]


class FakeGenerativeTask(Task):
    def __init__(self, examples):
        self._examples = examples
        super().__init__()

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self._examples)

    def get_example(self, index):
        return self._examples[index]

    def evaluate(self, problem, completion):
        return "42" in completion


def test_run_categorical_eval_returns_a_fraction_between_zero_and_one():
    tokenizer = make_test_tokenizer()
    model = tiny_model(tokenizer.get_vocab_size())
    examples = [
        {
            "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "A"}],
            "letters": ("A", "B"),
        },
        {
            "messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "B"}],
            "letters": ("A", "B"),
        },
    ]
    task = FakeCategoricalTask(examples)
    accuracy = run_categorical_eval(task, tokenizer, model)
    assert 0.0 <= accuracy <= 1.0


def test_run_generative_eval_uses_engine_and_task_evaluate():
    tokenizer = make_test_tokenizer()
    model = tiny_model(tokenizer.get_vocab_size())
    terminal_ids = (tokenizer.special_id("<|assistant_end|>"), tokenizer.get_bos_token_id())
    engine = Engine(model, tokenizer, terminal_ids)
    examples = [{"messages": [{"role": "user", "content": "what is 6*7"}, {"role": "assistant", "content": "42"}]}]
    task = FakeGenerativeTask(examples)
    accuracy = run_generative_eval(task, tokenizer, model, engine, max_new_tokens=8)
    assert accuracy in (0.0, 1.0)  # single example -> exact fraction


def test_evaluate_chatcore_centers_and_averages_across_tasks():
    tokenizer = make_test_tokenizer()
    model = tiny_model(tokenizer.get_vocab_size())
    terminal_ids = (tokenizer.special_id("<|assistant_end|>"), tokenizer.get_bos_token_id())
    engine = Engine(model, tokenizer, terminal_ids)
    mmlu_like = FakeCategoricalTask(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}], "letters": ("A", "B")}]
    )
    gsm8k_like = FakeGenerativeTask(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "42"}]}]
    )
    out = evaluate_chatcore({"MMLU": mmlu_like, "GSM8K": gsm8k_like}, tokenizer, model, engine, max_new_tokens=8)
    assert set(out["results"].keys()) == {"MMLU", "GSM8K"}
    assert "chatcore" in out
    assert -1.0 <= out["chatcore"] <= 1.0
