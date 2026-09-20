"""Tests for chat_sft.py's render_and_pack: BOS-aligned best-fit packing.
Uses the same in-memory byte-level tokenizer fixture as test_chat_tokenizer.py
and a tiny synthetic conversation list -- no network, no real task data."""
import mlx.core as mx
import tiktoken

from chat_sft import render_and_pack
from chat_tokenizer import ChatTokenizer, CHAT_SPECIAL_TOKENS


def make_test_tokenizer():
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    special_tokens = {name: 256 + i for i, name in enumerate(CHAT_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="test", pat_str=r".", mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
    return ChatTokenizer(enc)


class ListDataset:
    def __init__(self, conversations):
        self._conversations = conversations

    def __len__(self):
        return len(self._conversations)

    def __getitem__(self, index):
        return self._conversations[index % len(self._conversations)]


def make_conversation(user_text, assistant_text):
    return {"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]}


def test_packed_rows_have_correct_shape():
    tokenizer = make_test_tokenizer()
    dataset = ListDataset([make_conversation("hi", "hello"), make_conversation("bye", "goodbye")])
    loader = render_and_pack(tokenizer, dataset, batch_size=2, seq_len=32)
    inputs, targets, epoch = next(loader)
    assert inputs.shape == (2, 32)
    assert targets.shape == (2, 32)
    assert epoch >= 1


def test_masked_positions_become_ignore_index_minus_one():
    tokenizer = make_test_tokenizer()
    dataset = ListDataset([make_conversation("hi", "hello")])
    loader = render_and_pack(tokenizer, dataset, batch_size=1, seq_len=32)
    _, targets, _ = next(loader)
    values = targets.reshape(-1).tolist()
    assert -1 in values, "expected at least some masked (-1) positions (user turn / padding)"
    assert any(v != -1 for v in values), "expected at least some supervised positions (assistant turn)"


def test_padding_uses_bos_when_nothing_fits():
    tokenizer = make_test_tokenizer()
    # A single, short conversation repeated: after packing what fits, the
    # remainder of each row must be BOS-padded with mask=0 (-> target -1),
    # never left as an arbitrary/garbage token id.
    dataset = ListDataset([make_conversation("hi", "yo")])
    loader = render_and_pack(tokenizer, dataset, batch_size=1, seq_len=64)
    inputs, targets, _ = next(loader)
    # Whatever the tail padding is, it must be all BOS in the inputs and all
    # -1 in the targets simultaneously (never one without the other).
    bos_id = tokenizer.get_bos_token_id()
    input_list, target_list = inputs.reshape(-1).tolist(), targets.reshape(-1).tolist()
    tail_start = None
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] == bos_id and target_list[i] == -1:
            tail_start = i
        else:
            break
    assert tail_start is not None, "expected some trailing BOS-padded, masked-out region"


def test_loader_never_crashes_across_many_batches():
    tokenizer = make_test_tokenizer()
    dataset = ListDataset([make_conversation(f"question {i}", f"answer {i}") for i in range(5)])
    loader = render_and_pack(tokenizer, dataset, batch_size=2, seq_len=32)
    for _ in range(10):
        inputs, targets, epoch = next(loader)
        assert inputs.shape == (2, 32)
        assert isinstance(epoch, int)
