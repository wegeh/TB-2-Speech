"""Dataset and data loading utilities."""

from __future__ import annotations

import math
import re
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
TEXT_COL_CANDIDATES = ["text", "transcript", "transcription", "sentence", "label", "target"]
FILE_COL_CANDIDATES = [
    "file",
    "filename",
    "path",
    "audio", 
    "wav",
    "utt",
    "utterance_id",
    "sentenceid",
    "id",
]


def _find_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        # exact match
        if cand_lower in lowered:
            return lowered[cand_lower]
        # partial match
        for col_lower, original in lowered.items():
            if cand_lower in col_lower:
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


def prepare_metadata(
    transcript_path: Path,
    audio_dir: Path,
    text_column: str | None = None,
    filename_column: str | None = None,
) -> pd.DataFrame:
    """Load transcript_clean.csv and attach audio paths plus metadata."""
    df = pd.read_csv(transcript_path)
    file_col = (
        filename_column
        if filename_column and filename_column in df.columns
        else _find_column(df, FILE_COL_CANDIDATES, "audio filename")
    )
    text_col = (
        text_column
        if text_column and text_column in df.columns
        else _find_column(df, TEXT_COL_CANDIDATES, "text")
    )

    records = []
    for _, row in df.iterrows(): 
        filename = str(row[file_col])
        if not filename.lower().endswith(".wav"):
            filename = f"{filename}.wav"
        filepath = audio_dir / filename
        # Handle cases where transcript uses utt1 instead of utt01
        if not filepath.exists():
            match = re.search(r"(utt)(\d+)(\.wav)$", filename, flags=re.IGNORECASE)
            if match:
                prefix, num, suffix = match.groups()
                if len(num) == 1:
                    alt_filename = re.sub(
                        r"(utt)\d+(\.wav)$",
                        rf"\1{num.zfill(2)}\2",
                        filename,
                        flags=re.IGNORECASE,
                    )
                    alt_path = audio_dir / alt_filename
                    if alt_path.exists():
                        filename = alt_filename
                        filepath = alt_path
        try:
            speaker_id, native_flag = parse_filename_metadata(filename)
        except ValueError as exc:
            print(f"Skipping row due to filename parse error: {filename} ({exc})")
            continue
        if not filepath.exists():
            print(f"Skipping missing audio file: {filepath}")
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


def _sample_speakers_fixed(speakers: List[str], k: int, rng: random.Random) -> List[str]:
    """Sample up to k speakers (at least 1 if available)."""
    if not speakers:
        return []
    k = max(1, min(len(speakers), k))
    return rng.sample(speakers, k)


def create_splits(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create splits with speaker-disjoint test and per-utterance train/val:
    - Test: fixed 7 native speakers + 7 non-native speakers (by speaker, disjoint; capped by availability).
    - Train/Val: on remaining utterances, split per-utterance to target 70%/10% of total data.
    """
    rng = random.Random(seed)
    df = df.copy()

    native_speakers = sorted(set(df[df["native_flag"] == "n"]["speaker_id"]))
    non_native_speakers = sorted(set(df[df["native_flag"] == "nn"]["speaker_id"]))

    test_native = _sample_speakers_fixed(native_speakers, 7, rng)
    test_non_native = _sample_speakers_fixed(non_native_speakers, 7, rng)
    test_speakers = set(test_native + test_non_native)

    df["split"] = "train"

    if test_speakers:
        df.loc[df["speaker_id"].isin(test_speakers), "split"] = "test"

    remaining_df = df[df["split"] != "test"]

    train_keep = []
    speaker_groups = remaining_df.groupby("speaker_id")
    for _, group in speaker_groups:
        indices = sorted(group.index.tolist())
        rng.shuffle(indices)
        train_keep.append(indices[0])

    leftover_indices = set(remaining_df.index.tolist()) - set(train_keep)
    leftover_indices = sorted(list(leftover_indices))
    rng.shuffle(leftover_indices)

    target_val = int(round(len(df) * 0.10))
    val_count = min(len(leftover_indices), target_val)
    val_indices = set(leftover_indices[:val_count])
    if val_indices:
        df.loc[df.index.isin(val_indices), "split"] = "val"

    return {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }


class JavaneseDataset(Dataset):
    """
    Dataset for Javanese ASR.
    """
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


def collate_fn(batch: List[Dict], blank_id: int = 0) -> Dict:
    """
    Collate batch of examples.
    """
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
