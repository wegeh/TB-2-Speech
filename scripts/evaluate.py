"""
Evaluate scratch and fine-tuned models on the held-out test set.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
import soundfile as sf
import subprocess
import yaml
from jiwer import cer, wer
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import DEFAULT_VOCAB, JavaneseDataset, collate_fn, create_splits, prepare_metadata
from src.model import ConformerCTC
from src.utils import greedy_decoder, set_all_seeds


def load_audio_fallback(path: Path, target_sample_rate: int) -> torch.Tensor:
    """
    Load audio with torchaudio, fallback to soundfile, then FFmpeg CLI if needed.
    Returns mono waveform (tensor) at target_sample_rate.
    """
    path = Path(path)
    waveform = None
    sr = None

    try:
        waveform, sr = torchaudio.load(path)
    except (RuntimeError, OSError):
        pass

    if waveform is None:
        try:
            data, sr = sf.read(str(path))
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.transpose(0, 1)
        except Exception:
            waveform = None

    if waveform is None:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-ar",
                    str(target_sample_rate),
                    "-ac",
                    "1",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            data, sr = sf.read(str(path))
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.transpose(0, 1)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio {path}: {exc}") from exc

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sr, target_sample_rate)(waveform)
    return waveform.squeeze(0)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_scratch(
    model: ConformerCTC,
    dataloader,
    device: torch.device,
    idx_to_char: List[str],
) -> Dict:
    model.eval()
    records = []
    refs = []
    hyps = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scratch eval"):
            waveforms = batch["waveforms"].to(device)
            waveform_lengths = batch["waveform_lengths"].to(device)
            log_probs, _ = model(waveforms, waveform_lengths)
            preds = greedy_decoder(log_probs, idx_to_char, blank_id=0)
            refs.extend(batch["transcripts"])
            hyps.extend(preds)
            for fname, ref, pred in zip(batch["filenames"], batch["transcripts"], preds):
                records.append(
                    {
                        "filename": fname,
                        "target": ref,
                        "prediction_scratch": pred,
                    }
                )

    return {
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "records": records,
    }


def evaluate_finetune(
    model_dir: Path,
    test_df: pd.DataFrame,
    sample_rate: int,
    device: torch.device,
) -> Dict[str, str]:
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    preds = {}
    refs = []
    hyps = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finetune eval"):
        wav_path = Path(row["filepath"])
        waveform = load_audio_fallback(wav_path, sample_rate)
        input_values = processor(
            waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt"
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0].lower().strip()

        preds[wav_path.name] = pred_text
        refs.append(row["transcript"])
        hyps.append(pred_text)

    return {"preds": preds, "wer": wer(refs, hyps), "cer": cer(refs, hyps)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate scratch vs fine-tuned models.")
    parser.add_argument(
        "--scratch_config",
        type=Path,
        default=Path("configs/config_scratch.yaml"),
        help="Config used for scratch model.",
    )
    parser.add_argument(
        "--scratch_checkpoint",
        type=Path,
        default=Path("checkpoints/scratch/best_model.pth"),
        help="Path to scratch model checkpoint (.pth).",
    )
    parser.add_argument(
        "--finetune_dir",
        type=Path,
        default=Path("checkpoints/finetune"),
        help="Directory containing fine-tuned HF model.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Path to save predictions CSV.",
    )
    return parser.parse_args()


def main():
    set_all_seeds(42)
    args = parse_args()
    scratch_cfg = load_yaml(args.scratch_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    data_cfg = scratch_cfg.get("data", {})
    model_cfg = scratch_cfg.get("model", {})
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    transcript_path = Path(data_cfg.get("transcript_path", "data/transcript_clean.csv"))
    audio_dir = Path(data_cfg.get("audio_dir", "data/wav_clean"))
    seed = int(data_cfg.get("seed", 42))

    metadata = prepare_metadata(transcript_path, audio_dir)
    splits = create_splits(metadata, seed=seed)
    test_df = splits["test"]

    vocab = DEFAULT_VOCAB
    idx_to_char = vocab
    blank_id = 0

    test_dataset = JavaneseDataset(test_df, vocab=vocab, sample_rate=sample_rate)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, blank_id=blank_id),
    )

    # Scratch model
    scratch_model = ConformerCTC(
        vocab_size=len(vocab),
        encoder_dim=int(model_cfg.get("encoder_dim", 256)),
        encoder_layers=int(model_cfg.get("encoder_layers", 12)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        n_mels=int(model_cfg.get("n_mels", 80)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        sample_rate=sample_rate,
    ).to(device)

    if args.scratch_checkpoint.exists():
        state = torch.load(args.scratch_checkpoint, map_location=device)
        scratch_model.load_state_dict(state["model_state"])
        print(f"Loaded scratch checkpoint: {args.scratch_checkpoint}")
    else:
        print(f"Warning: scratch checkpoint not found at {args.scratch_checkpoint}")

    scratch_metrics = evaluate_scratch(scratch_model, test_loader, device, idx_to_char)

    # Fine-tuned model
    finetune_metrics = evaluate_finetune(args.finetune_dir, test_df, sample_rate, device)

    # Merge predictions
    finetune_preds = finetune_metrics["preds"]
    rows = []
    for rec in scratch_metrics["records"]:
        fname = rec["filename"]
        rows.append(
            {
                "filename": fname,
                "target": rec["target"],
                "prediction_scratch": rec["prediction_scratch"],
                "prediction_finetune": finetune_preds.get(fname, ""),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved prediction comparison to {args.output_csv}")
    print(
        f"Scratch WER: {scratch_metrics['wer']:.4f}, CER: {scratch_metrics['cer']:.4f}\n"
        f"Finetune WER: {finetune_metrics['wer']:.4f}, CER: {finetune_metrics['cer']:.4f}"
    )


if __name__ == "__main__":
    main()
