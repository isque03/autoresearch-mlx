"""Tests for chat_tokenizer.py's render_conversation/render_for_completion.
Uses a minimal in-memory tiktoken.Encoding (byte-level, no real BPE merges)
so these tests need no network access, no downloaded data, and no
prepare_chat.py run -- fast and fully deterministic."""
import tiktoken

from chat_tokenizer import ChatTokenizer, CHAT_SPECIAL_TOKENS, BOS_TOKEN


def make_test_tokenizer() -> ChatTokenizer:
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    special_tokens = {name: 256 + i for i, name in enumerate(CHAT_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="test-byte-level", pat_str=r".", mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return ChatTokenizer(enc)


def simple_conversation(*turns):
    """turns: alternating (role, content) pairs, e.g. ("user","hi"), ("assistant","hello")."""
    return {"messages": [{"role": role, "content": content} for role, content in turns]}


def test_starts_with_bos_mask_zero():
    tok = make_test_tokenizer()
    ids, mask = tok.render_conversation(simple_conversation(("user", "hi"), ("assistant", "yo")))
    assert ids[0] == tok.get_bos_token_id()
    assert mask[0] == 0


def test_user_tokens_all_masked_out():
    tok = make_test_tokenizer()
    ids, mask = tok.render_conversation(simple_conversation(("user", "hello there"), ("assistant", "hi")))
    user_start = tok.special_id("<|user_start|>")
    user_end = tok.special_id("<|user_end|>")
    start_idx = ids.index(user_start)
    end_idx = ids.index(user_end)
    assert all(m == 0 for m in mask[start_idx:end_idx + 1])


def test_assistant_content_and_end_are_masked_in():
    tok = make_test_tokenizer()
    ids, mask = tok.render_conversation(simple_conversation(("user", "hi"), ("assistant", "hello")))
    assistant_start = tok.special_id("<|assistant_start|>")
    assistant_end = tok.special_id("<|assistant_end|>")
    start_idx = ids.index(assistant_start)
    end_idx = ids.index(assistant_end)
    # assistant_start itself is not supervised (mask=0), but everything from
    # just after it through assistant_end (inclusive) is (mask=1).
    assert mask[start_idx] == 0
    assert all(m == 1 for m in mask[start_idx + 1:end_idx + 1])


def test_system_message_spliced_into_first_user_turn():
    tok = make_test_tokenizer()
    conversation = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    }
    ids, mask = tok.render_conversation(conversation)
    # No dedicated system token exists -- splice means the rendered sequence
    # is identical in *shape* to the same conversation with the system text
    # prepended directly onto the user message.
    spliced = simple_conversation(("user", "be terse\n\nhi"), ("assistant", "yo"))
    expected_ids, expected_mask = tok.render_conversation(spliced)
    assert ids == expected_ids
    assert mask == expected_mask


def test_tool_invocation_is_supervised_tool_output_is_not():
    tok = make_test_tokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "what is 2+2"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me compute"},
                    {"type": "python", "text": "2+2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "the answer is 4"},
                ],
            },
        ]
    }
    ids, mask = tok.render_conversation(conversation)
    python_start = tok.special_id("<|python_start|>")
    output_start = tok.special_id("<|output_start|>")
    output_end = tok.special_id("<|output_end|>")

    py_idx = ids.index(python_start)
    assert mask[py_idx] == 1  # invocation IS supervised

    out_start_idx = ids.index(output_start)
    out_end_idx = ids.index(output_end)
    assert all(m == 0 for m in mask[out_start_idx:out_end_idx + 1])  # output NOT supervised


def test_strict_alternation_enforced():
    tok = make_test_tokenizer()
    bad = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    try:
        tok.render_conversation(bad)
        assert False, "expected an assertion error for non-alternating roles"
    except AssertionError:
        pass


def test_truncates_to_max_tokens():
    tok = make_test_tokenizer()
    long_text = "x" * 500
    ids, mask = tok.render_conversation(
        simple_conversation(("user", long_text), ("assistant", long_text)), max_tokens=10
    )
    assert len(ids) == 10
    assert len(mask) == 10


def test_render_for_completion_primes_with_assistant_start_and_no_ground_truth():
    tok = make_test_tokenizer()
    conversation = simple_conversation(("user", "hi"), ("assistant", "this should not appear"))
    ids = tok.render_for_completion(conversation)
    assert ids[-1] == tok.special_id("<|assistant_start|>")
    decoded_tail = tok.decode(ids)
    assert "this should not appear" not in decoded_tail
