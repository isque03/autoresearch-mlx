"""Chat tokenizer + template: wraps chat_tokenizer.pkl (trained by
prepare_chat.py) and implements the conversation -> (ids, mask) rendering
that chat_sft.py trains on and chat_cli.py/chat_eval.py prime generation
with. Mirrors nanochat's nanochat/tokenizer.py::render_conversation exactly
(same special-token set, same system-message splice, same mask semantics)
so the SFT masking properties this repo tests for are the ones that matter
in practice, not an ad-hoc simplification.
"""
from __future__ import annotations

import copy
import pickle

from prepare_chat import CHAT_SPECIAL_TOKENS, CHAT_TOKENIZER_PKL, BOS_TOKEN

MAX_CONVERSATION_TOKENS = 2048


class ChatTokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)
        self._special_ids = {name: enc.encode_single_token(name) for name in CHAT_SPECIAL_TOKENS}

    @classmethod
    def from_directory(cls):
        with open(CHAT_TOKENIZER_PKL, "rb") as handle:
            enc = pickle.load(handle)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def special_id(self, name: str) -> int:
        return self._special_ids[name]

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def render_conversation(
        self, conversation: dict, max_tokens: int = MAX_CONVERSATION_TOKENS
    ) -> tuple[list[int], list[int]]:
        """Returns (ids, mask). mask[i]==1 means position i is a token the
        assistant should have generated and is trained on as a prediction
        target (ids/mask are consumed shifted-by-1 by the caller, same as
        pretrain.make_dataloader's inputs/targets split)."""
        messages = list(conversation["messages"])
        if messages and messages[0]["role"] == "system":
            assert len(messages) > 1 and messages[1]["role"] == "user", (
                "a leading system message must be followed by a user message"
            )
            messages[1] = {
                **messages[1],
                "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
            }
            messages = messages[1:]

        ids: list[int] = [self.bos_token_id]
        mask: list[int] = [0]
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"expected strictly alternating user/assistant starting with user, "
                f"got {message['role']!r} at position {i}"
            )
            if message["role"] == "user":
                assert isinstance(message["content"], str), "user content must be a plain string"
                ids.append(self.special_id("<|user_start|>"))
                mask.append(0)
                encoded = self.encode(message["content"])
                ids.extend(encoded)
                mask.extend([0] * len(encoded))
                ids.append(self.special_id("<|user_end|>"))
                mask.append(0)
            else:
                ids.append(self.special_id("<|assistant_start|>"))
                mask.append(0)
                content = message["content"]
                if isinstance(content, str):
                    encoded = self.encode(content)
                    ids.extend(encoded)
                    mask.extend([1] * len(encoded))
                else:
                    for part in content:
                        if part["type"] == "text":
                            encoded = self.encode(part["text"])
                            ids.extend(encoded)
                            mask.extend([1] * len(encoded))
                        elif part["type"] == "python":
                            encoded = self.encode(part["text"])
                            ids.append(self.special_id("<|python_start|>"))
                            ids.extend(encoded)
                            ids.append(self.special_id("<|python_end|>"))
                            mask.extend([1] * (len(encoded) + 2))
                        elif part["type"] == "python_output":
                            encoded = self.encode(part["text"])
                            ids.append(self.special_id("<|output_start|>"))
                            ids.extend(encoded)
                            ids.append(self.special_id("<|output_end|>"))
                            mask.extend([0] * (len(encoded) + 2))
                        else:
                            raise ValueError(f"unknown assistant content part type: {part['type']!r}")
                ids.append(self.special_id("<|assistant_end|>"))
                mask.append(1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation: dict) -> list[int]:
        """Renders everything except the final assistant message (removed),
        then primes generation with <|assistant_start|>. Used at inference
        time -- no mask needed since nothing here is a training target."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "last message must be the assistant turn to predict"
        conversation["messages"] = messages[:-1]
        ids, _ = self.render_conversation(conversation)
        ids.append(self.special_id("<|assistant_start|>"))
        return ids
