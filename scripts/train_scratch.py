"""
Train Conformer CTC from scratch using cleaned data.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import DEFAULT_VOCAB, JavaneseDataset, collate_fn, create_splits, prepare_metadata
from src.model import ConformerCTC
from src.trainer import CTCTrainer
from src.utils import set_all_seeds


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conformer ASR from scratch.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_scratch.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main():
    set_all_seeds(42)
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    transcript_path = Path(data_cfg.get("transcript_path", "data/transcript_clean.csv"))
    audio_dir = Path(data_cfg.get("audio_dir", "data/wav_clean"))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    seed = int(data_cfg.get("seed", 42))

    metadata = prepare_metadata(transcript_path, audio_dir)
    splits = create_splits(metadata, seed=seed)

    vocab = DEFAULT_VOCAB
    idx_to_char = vocab
    blank_id = 0

    train_dataset = JavaneseDataset(splits["train"], vocab=vocab, sample_rate=sample_rate)
    val_dataset = JavaneseDataset(splits["val"], vocab=vocab, sample_rate=sample_rate)

    num_workers = int(data_cfg.get("num_workers", 0))
    collate = functools.partial(collate_fn, blank_id=blank_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )

    model = ConformerCTC(
        vocab_size=len(vocab),
        encoder_dim=int(model_cfg.get("encoder_dim", 256)),
        encoder_layers=int(model_cfg.get("encoder_layers", 12)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        n_mels=int(model_cfg.get("n_mels", 80)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        sample_rate=sample_rate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None

    trainer = CTCTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        idx_to_char=idx_to_char,
        blank_id=blank_id,
        grad_clip=float(train_cfg.get("grad_clip", 5.0)),
        log_interval=int(train_cfg.get("log_interval", 10)),
    )

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/scratch"))
    epochs = int(train_cfg.get("epochs", 50))

    trainer.fit(train_loader, val_loader, epochs=epochs, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
