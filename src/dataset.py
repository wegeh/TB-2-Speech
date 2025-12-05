"""Dataset utilities for Javanese ASR."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torchaudio
import soundfile as sf
import subprocess
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEFAULT_VOCAB = ["<blank>"] + list(" abcdefghijklmnopqrstuvwxyz'")
TEXT_COL_CANDIDATES = ["text", "transcript", "sentence", "label", "target"]
FILE_COL_CANDIDATES = ["file", "filename", "path", "audio", "wav", "utt", "utterance_id"]


def _find_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for col_lower, original in lowered.items():
        for cand in candidates:
            if cand == col_lower or cand in col_lower:
                return original
    raise ValueError(
        f"Cannot find {kind} column. Looked for {candidates}. Available: {list(df.columns)}"
    )


def parse_filename_metadata(filename: str) -> Tuple[str, str]:
    """Parse speaker_id and native/non-native flag (n/nn) from filename."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    speaker_id = parts[0].lower()
    native_flag = parts[2].lower()
    native_flag = "n" if native_flag == "n" else "nn"
    return speaker_id, native_flag


def prepare_metadata(transcript_path: Path, audio_dir: Path) -> pd.DataFrame:
    """Load transcript_clean.csv and attach audio paths plus metadata."""
    df = pd.read_csv(transcript_path)
    file_col = _find_column(df, FILE_COL_CANDIDATES, "audio filename")
    text_col = _find_column(df, TEXT_COL_CANDIDATES, "text")

    records = []
    for _, row in df.iterrows():
        filename = str(row[file_col])
        if not filename.lower().endswith(".wav"):
            filename = f"{filename}.wav"
        filepath = audio_dir / filename
        try:
            speaker_id, native_flag = parse_filename_metadata(filename)
        except ValueError as exc:
            print(f"Skipping row due to filename parse error: {filename} ({exc})")
            continue
        records.append(
            {
                "filepath": filepath,
                "transcript": row[text_col],
                "speaker_id": speaker_id,
                "native_flag": native_flag,
            }
        )
    return pd.DataFrame(records)


def _sample_speakers(speakers: List[str], fraction: float, rng: random.Random) -> List[str]:
    if not speakers:
        return []
    k = max(1, int(math.ceil(len(speakers) * fraction)))
    k = min(len(speakers), k)
    return rng.sample(speakers, k)


def create_splits(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create speaker-disjoint splits:
    - Test: 10% native speakers + 10% non-native speakers (20% total speakers).
    - Remaining speakers split into Train/Val with 70/10 proportion of total data.
    """
    rng = random.Random(seed)
    df = df.copy()

    native_speakers = sorted(set(df[df["native_flag"] == "n"]["speaker_id"]))
    non_native_speakers = sorted(set(df[df["native_flag"] == "nn"]["speaker_id"]))

    test_native = _sample_speakers(native_speakers, 0.10, rng)
    test_non_native = _sample_speakers(non_native_speakers, 0.10, rng)
    test_speakers = set(test_native + test_non_native)

    remaining_speakers = [s for s in sorted(set(df["speaker_id"])) if s not in test_speakers]
    rng.shuffle(remaining_speakers)

    # To reach 70/10 train/val overall after removing 20% test: 87.5% / 12.5% of remaining.
    if remaining_speakers:
        val_count = max(1, int(round(len(remaining_speakers) * 0.125)))
        if len(remaining_speakers) - val_count <= 0:
            val_count = max(0, len(remaining_speakers) - 1)
    else:
        val_count = 0
    val_speakers = set(remaining_speakers[:val_count])
    train_speakers = set(remaining_speakers[val_count:])

    df["split"] = "train"
    if val_speakers:
        df.loc[df["speaker_id"].isin(val_speakers), "split"] = "val"
    if test_speakers:
        df.loc[df["speaker_id"].isin(test_speakers), "split"] = "test"

    return {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }


def encode_text(text: str, vocab: List[str]) -> List[int]:
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


class JavaneseDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        vocab: List[str] | None = None,
        sample_rate: int = 16000,
    ):
        self.data = data.reset_index(drop=True)
        self.vocab = vocab or DEFAULT_VOCAB
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        wav_path = Path(row["filepath"])
        waveform = None
        sr = None

        # Try torchaudio first.
        try:
            waveform, sr = torchaudio.load(wav_path)
        except (RuntimeError, OSError):
            pass

        # Fallback to soundfile.
        if waveform is None:
            try:
                data, sr = sf.read(str(wav_path))
                waveform = torch.tensor(data, dtype=torch.float32)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                    waveform = waveform.transpose(0, 1)
            except Exception:
                waveform = None

        # Last resort: ffmpeg CLI decode then load via soundfile.
        if waveform is None:
            tmp_path = wav_path
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(wav_path),
                        "-ar",
                        str(self.sample_rate),
                        "-ac",
                        "1",
                        str(wav_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                data, sr = sf.read(str(wav_path))
                waveform = torch.tensor(data, dtype=torch.float32)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                    waveform = waveform.transpose(0, 1)
            except Exception as exc:
                raise RuntimeError(f"Failed to load audio {wav_path}: {exc}") from exc

        # Ensure mono and resample
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.squeeze(0)

        transcript = row["transcript"]
        labels = torch.tensor(
            [self.char_to_idx[ch] for ch in transcript if ch in self.char_to_idx],
            dtype=torch.long,
        )

        return {
            "waveform": waveform,
            "waveform_length": waveform.shape[-1],
            "labels": labels,
            "label_length": len(labels),
            "transcript": transcript,
            "speaker_id": row["speaker_id"],
            "native_flag": row["native_flag"],
            "filename": wav_path.name,
        }


def collate_fn(batch: List[Dict], blank_id: int = 0) -> Dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["labels"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    filenames = [item.get("filename") for item in batch]
    speaker_ids = [item.get("speaker_id") for item in batch]
    native_flags = [item.get("native_flag") for item in batch]

    waveforms = pad_sequence(waveforms, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=blank_id)

    waveform_lengths = torch.tensor([item["waveform_length"] for item in batch], dtype=torch.long)
    label_lengths = torch.tensor([item["label_length"] for item in batch], dtype=torch.long)

    return {
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths,
        "transcripts": transcripts,
        "filenames": filenames,
        "speaker_ids": speaker_ids,
        "native_flags": native_flags,
    }
