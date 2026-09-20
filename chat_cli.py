"""Interactive chat CLI over an SFT checkpoint. Mirrors nanochat's
scripts/chat_cli.py shape: load model + chat tokenizer, maintain a running
conversation, generate one assistant turn per user input via the KV-cache
Engine.

Usage:
    python chat_cli.py [chatsft_checkpoints]
"""
import sys

from chat_tokenizer import ChatTokenizer
from checkpoint_manager import build_model_from_checkpoint
from engine import Engine
from train import GPT, GPTConfig

CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else "chatsft_checkpoints"


def main():
    tokenizer = ChatTokenizer.from_directory()
    model, metadata = build_model_from_checkpoint(CHECKPOINT_DIR, GPT, GPTConfig)
    print(f"Loaded {CHECKPOINT_DIR} (step={metadata.get('step')})", flush=True)

    terminal_ids = (tokenizer.special_id("<|assistant_end|>"), tokenizer.get_bos_token_id())
    engine = Engine(model, tokenizer, terminal_ids)

    messages = []
    print("Chat started. Ctrl-C to exit.", flush=True)
    while True:
        try:
            user_text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        messages.append({"role": "user", "content": user_text})
        conversation = {"messages": messages + [{"role": "assistant", "content": ""}]}
        prompt_ids = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(prompt_ids, num_samples=1, max_tokens=256, temperature=0.0)
        completion_ids = results[0][len(prompt_ids):]
        response = tokenizer.decode(completion_ids)
        print(response, flush=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
