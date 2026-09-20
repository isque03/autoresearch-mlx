"""One-time setup for the chat-capable tokenizer used by chat_sft.py/chat_eval.py/
chat_cli.py. Reuses the exact same BPE merges prepare.py trains (same corpus,
same VOCAB_SIZE, same SPLIT_PATTERN) so a pretrain checkpoint's embedding table
stays semantically valid when warm-started into SFT -- only the special-token
slots differ (9 chat-control tokens instead of prepare.py's 4 reserved slots,
1 of which it uses for BOS). Written as a separate artifact (chat_tokenizer.pkl)
so prepare.py's own tokenizer.pkl (the pretrain loop's fixed contract) is never
touched.

Usage:
    python prepare_chat.py
"""
import os
import pickle
import time

import tiktoken

from prepare import (
    SPLIT_PATTERN, TOKENIZER_DIR, VOCAB_SIZE, list_parquet_files, text_iterator,
)

CHAT_SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]
BOS_TOKEN = "<|bos|>"
CHAT_TOKENIZER_PKL = os.path.join(TOKENIZER_DIR, "chat_tokenizer.pkl")


def train_chat_tokenizer():
    if os.path.exists(CHAT_TOKENIZER_PKL):
        print(f"Chat tokenizer: already trained at {CHAT_TOKENIZER_PKL}")
        return

    if len(list_parquet_files()) < 2:
        raise SystemExit("Chat tokenizer: need pretrain data shards first. Run prepare.py.")

    import rustbpe

    print("Chat tokenizer: training BPE merges (same corpus/vocab as prepare.py)...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(CHAT_SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(CHAT_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe-chat", pat_str=pattern, mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    with open(CHAT_TOKENIZER_PKL, "wb") as handle:
        pickle.dump(enc, handle)
    print(f"Chat tokenizer: trained in {time.time() - t0:.1f}s, saved to {CHAT_TOKENIZER_PKL}")
    print(f"Chat tokenizer: vocab_size={enc.n_vocab}, special_tokens={CHAT_SPECIAL_TOKENS}")


if __name__ == "__main__":
    train_chat_tokenizer()
    print("Done! Ready for chat_sft.py.")
