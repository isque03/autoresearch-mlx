"""SmolTalk: general multi-turn conversational SFT data. Mirrors nanochat's
tasks/smoltalk.py -- pure passthrough of the HF dataset's own message list,
after validating the alternation/system-message shape chat_tokenizer.py
expects. No eval_type/evaluate -- used only for SFT training, never scored."""
from __future__ import annotations

from tasks.common import HubDataset, Task


def _validate_messages(messages: list[dict]) -> None:
    start = 1 if messages and messages[0]["role"] == "system" else 0
    for i, message in enumerate(messages[start:]):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected_role, f"expected alternating roles starting with user, got {messages}"
        assert isinstance(message["content"], str), "smoltalk messages must have plain string content"


class SmolTalk(Task):
    def __init__(self, split: str = "train", **kwargs):
        assert split in ("train", "test")
        self._data = HubDataset("HuggingFaceTB/smol-smoltalk", subset="default", split=split)
        super().__init__(**kwargs)

    @property
    def eval_type(self) -> str:
        return "none"

    def num_examples(self) -> int:
        return len(self._data)

    def get_example(self, index: int) -> dict:
        row = self._data[index]
        messages = row["messages"]
        _validate_messages(messages)
        return {"messages": messages}
