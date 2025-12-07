"""
Data cleaning utilities for the Javanese ASR project.

Functions included:
- Text cleaning to lowercase, remove punctuation (except apostrophes), and drop digits.
- Audio resampling to 16 kHz mono.
"""

from __future__ import annotations

import argparse
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio

PUNCT_TO_REMOVE = string.punctuation.replace("'", "")
TEXT_COL_CANDIDATES = ["text", "transcript", "sentence", "label", "target"]
FILE_COL_CANDIDATES = [
    "file",
    "filename",
    "path",
    "audio",
    "wav",
    "utt",
    "utterance_id",
]


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation (keep apostrophes) and digits, collapse whitespace."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    # Remove all punctuation including apostrophes
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    text = re.sub(r"[0-9]", " ", text)
    # Only allow a-z and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    candidates = list(candidates)
    lowered = {col.lower(): col for col in df.columns}

    for col_lower, original in lowered.items():
        for cand in candidates:
            if cand == col_lower or cand in col_lower:
                return original

    raise ValueError(
        f"Cannot find {kind} column. Looked for {candidates}. Available: {list(df.columns)}"
    )


def _guess_text_column(df: pd.DataFrame) -> str:
    """Pick a reasonable text column if no known name matches."""
    for col in df.columns:
        if df[col].dtype == object:
            return col
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str)).any():
            return col
    return df.columns[0]


def _attach_audio_filenames(
    df: pd.DataFrame,
    audio_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """If no filename column exists, align transcripts to audio files sorted by name."""
    audio_files = sorted(
        [p for p in audio_dir.rglob("*.wav") if p.is_file()]
        + [p for p in audio_dir.rglob("*.WAV") if p.is_file()]
    )
    if not audio_files:
        raise ValueError(
            f"No audio filename column and no .wav files found in {audio_dir} to align with."
        )
    if len(audio_files) < len(df):
        raise ValueError(
            f"Not enough audio files to align with transcripts. "
            f"Audio files: {len(audio_files)}, transcripts: {len(df)}."
        )
    df = df.copy()
    df["filename"] = [p.name for p in audio_files[: len(df)]]
    print(
        "Info: Audio filename column not found. "
        "Aligned transcripts to audio files in alphabetical order."
    )
    return df, "filename"


def load_transcript(
    transcript_path: Path, sheet_name: str | int | None = 0
) -> Tuple[pd.DataFrame, Path]:
    """Load transcript from CSV or XLSX. Returns dataframe and actual path used."""
    candidates = [transcript_path]
    if transcript_path.suffix == "":
        candidates.append(transcript_path.with_suffix(".csv"))
        candidates.append(transcript_path.with_suffix(".xlsx"))
    else:
        alt = transcript_path.with_suffix(
            ".xlsx" if transcript_path.suffix != ".xlsx" else ".csv"
        )
        candidates.append(alt)
    candidates.append(Path("data/transcripts.xlsx"))

    for candidate in candidates:
        if candidate.exists():
            sheet_to_use = 0 if sheet_name is None else sheet_name
            if isinstance(sheet_to_use, str) and sheet_to_use.isdigit():
                sheet_to_use = int(sheet_to_use)
            if candidate.suffix.lower() == ".xlsx":
                df = pd.read_excel(candidate, sheet_name=sheet_to_use)
                if isinstance(df, dict):
                    # Pick the first sheet if a dict is returned.
                    first_key = next(iter(df))
                    print(
                        f"Info: sheet_name returned multiple sheets; using first sheet: {first_key}"
                    )
                    df = df[first_key]
            else:
                df = pd.read_csv(candidate)
            return df, candidate

    raise FileNotFoundError(
        f"Transcript file not found. Tried: {[str(p) for p in candidates]}"
    )


def process_transcripts(
    transcript_path: Path,
    output_path: Path,
    audio_dir: Path | None = None,
    text_column: str | None = None,
    filename_column: str | None = None,
    sheet_name=None,
) -> Tuple[int, Path]:
    """Clean transcripts and write to CSV. Returns row count and source path."""
    df, used_path = load_transcript(transcript_path, sheet_name=sheet_name)
    if text_column is not None and text_column in df.columns:
        text_col = text_column
    else:
        try:
            text_col = _find_column(df, TEXT_COL_CANDIDATES, "text")
        except ValueError:
            text_col = _guess_text_column(df)

    if filename_column is not None and filename_column in df.columns:
        file_col = filename_column
    else:
        try:
            file_col = _find_column(df, FILE_COL_CANDIDATES, "audio filename")
        except ValueError:
            if audio_dir is None:
                raise
            df, file_col = _attach_audio_filenames(df, audio_dir)

    cleaned = df[[file_col, text_col]].copy()
    cleaned[text_col] = cleaned[text_col].apply(clean_text)
    cleaned = cleaned[cleaned[text_col].str.len() > 0]

    if len(cleaned) == 0:
        raise ValueError(
            "No transcripts remained after cleaning. "
            "Ensure the correct text column is selected (use --text_column) "
            "and the data contains non-empty text."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return len(cleaned), used_path


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform tensor to mono."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def process_audio_files(
    input_dir: Path,
    output_dir: Path,
    target_sample_rate: int = 16000,
) -> int:
    """Resample all WAV files to target_sample_rate mono. Returns file count processed."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input audio directory not found: {input_dir}")

    wav_files = sorted(
        [p for p in input_dir.rglob("*.wav") if p.is_file()]
        + [p for p in input_dir.rglob("*.WAV") if p.is_file()]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total = len(wav_files)
    for i, wav_path in enumerate(wav_files, start=1):
        print(f"[{i}/{total}] Processing {wav_path}")
        target_path = output_dir / wav_path.name

        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform = _to_mono(waveform)
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                waveform = resampler(waveform)
            try:
                torchaudio.save(target_path, waveform, target_sample_rate)
            except (RuntimeError, OSError):
                sf.write(target_path, waveform.squeeze(0).numpy(), target_sample_rate)
            processed += 1
            continue
        except (RuntimeError, OSError):
            pass

        # Fallback 1: soundfile decode
        try:
            data, sr = sf.read(str(wav_path))
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] > waveform.shape[0]:
                waveform = waveform.transpose(0, 1)
            waveform = _to_mono(waveform)
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                waveform = resampler(waveform)
            sf.write(target_path, waveform.squeeze(0).numpy(), target_sample_rate)
            processed += 1
            continue
        except Exception:
            pass

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-ar",
            str(target_sample_rate),
            "-ac",
            "1",
            str(target_path),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print(f"[WARN] ffmpeg not found, skipping {wav_path}")
            continue

        if result.returncode != 0:
            print(
                f"[WARN] Failed to convert {wav_path} via ffmpeg, skipping.\n"
                f"stderr: {result.stderr.decode(errors='ignore')[:200]}"
            )
            continue

        processed += 1

    return processed


def run_cleaning(
    transcript_path: Path,
    output_transcript: Path,
    audio_dir: Path,
    output_audio_dir: Path,
    sample_rate: int = 16000,
    text_column: str | None = None,
    filename_column: str | None = None,
    sheet_name=None,
) -> None:
    print(f"Cleaning transcripts from: {transcript_path}")
    rows, used_path = process_transcripts(
        transcript_path,
        output_transcript,
        audio_dir=audio_dir,
        text_column=text_column,
        filename_column=filename_column,
        sheet_name=sheet_name,
    )
    print(f" - Loaded transcript file: {used_path}")
    print(f" - Cleaned rows written to: {output_transcript} (rows: {rows})")

    print(f"Processing audio from: {audio_dir}")
    count = process_audio_files(
        audio_dir, output_audio_dir, target_sample_rate=sample_rate
    )
    print(f" - Processed {count} audio files to {output_audio_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean text transcripts and audio files."
    )
    parser.add_argument(
        "--transcript_path",
        type=Path,
        default=Path("data/transcript.csv"),
        help="Path to raw transcript CSV/XLSX.",
    )
    parser.add_argument(
        "--output_transcript",
        type=Path,
        default=Path("data/transcript_clean.csv"),
        help="Path to save cleaned transcript CSV.",
    )
    parser.add_argument(
        "--audio_dir",
        type=Path,
        default=Path("data/wav"),
        help="Directory containing raw WAV files.",
    )
    parser.add_argument(
        "--output_audio_dir",
        type=Path,
        default=Path("data/wav_clean"),
        help="Directory to save cleaned WAV files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for audio resampling.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Optional explicit text column name in transcript file.",
    )
    parser.add_argument(
        "--filename_column",
        type=str,
        default=None,
        help="Optional explicit filename column name in transcript file.",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        default="Data",
        help="Sheet name to read when using Excel transcripts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    audio_dir = args.audio_dir
    if not audio_dir.exists() and Path("data").exists():
        fallback_files = list(Path("data").glob("*.wav"))
        if fallback_files:
            print(f"Audio dir {audio_dir} not found; falling back to data/ root")
            audio_dir = Path("data")

    run_cleaning(
        transcript_path=args.transcript_path,
        output_transcript=args.output_transcript,
        audio_dir=audio_dir,
        output_audio_dir=args.output_audio_dir,
        sample_rate=args.sample_rate,
        text_column=args.text_column,
        filename_column=args.filename_column,
        sheet_name=args.sheet_name,
    )


if __name__ == "__main__":
    main()
