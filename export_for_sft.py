"""One-off: package the best existing pretrain checkpoint (saved by
train_extended.py's own ad-hoc {commit}_{val_bpb}.safetensors convention,
predating checkpoint_manager.py) into checkpoint_manager's directory+metadata
format, so chat_sft.py can warm-start from it. Reconstructs GPTConfig from
train.py's CURRENT architecture constants -- verified by hand to match the
commit that produced this checkpoint before running this.
"""
import glob
import os
import sys

from checkpoint_manager import load_weights_into, save_checkpoint
from prepare import Tokenizer
from train import (
    GPT, GPTConfig, ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, DEPTH,
    EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY,
)

SOURCE_DIR = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_extended"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "pretrain_for_sft"


def find_best_checkpoint(source_dir: str) -> str:
    candidates = glob.glob(os.path.join(source_dir, "*.safetensors"))
    if not candidates:
        raise SystemExit(f"no .safetensors files found in {source_dir}")
    return min(candidates, key=lambda p: float(os.path.basename(p).rsplit("_", 1)[1].removesuffix(".safetensors")))


def main():
    checkpoint_path = find_best_checkpoint(SOURCE_DIR)
    val_bpb = float(os.path.basename(checkpoint_path).rsplit("_", 1)[1].removesuffix(".safetensors"))
    print(f"Exporting {checkpoint_path} (val_bpb={val_bpb}) -> {OUTPUT_DIR}/", flush=True)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    config = GPTConfig(
        sequence_len=2048, vocab_size=vocab_size, n_layer=DEPTH,
        n_head=model_dim // HEAD_DIM, n_kv_head=model_dim // HEAD_DIM,
        n_embd=model_dim, window_pattern=WINDOW_PATTERN,
    )
    model = GPT(config)

    # load_weights_into expects a checkpoint DIRECTORY containing
    # model.safetensors; this source is a flat file, so load it directly.
    import mlx.core as mx
    from checkpoint_manager import set_path_value
    weights = mx.load(checkpoint_path)
    for path, value in weights.items():
        set_path_value(model, path, value)
    mx.eval(model.parameters())

    metadata = {
        "step": None,
        "val_bpb": val_bpb,
        "model_config": vars(config),
        "user_config": {
            "EMBEDDING_LR": EMBEDDING_LR, "UNEMBEDDING_LR": UNEMBEDDING_LR,
            "MATRIX_LR": MATRIX_LR, "SCALAR_LR": SCALAR_LR, "WEIGHT_DECAY": WEIGHT_DECAY,
        },
        "source_checkpoint": checkpoint_path,
    }
    save_checkpoint(OUTPUT_DIR, model, metadata)
    print(f"Exported to {OUTPUT_DIR}/ (model.safetensors + metadata.json)", flush=True)


if __name__ == "__main__":
    main()
