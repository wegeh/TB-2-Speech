"""Utility helpers: decoding and metrics."""

from __future__ import annotations

from typing import Iterable, List

import torch
from jiwer import cer, wer


def greedy_decoder(
    log_probs: torch.Tensor,
    idx_to_char: List[str],
    blank_id: int = 0,
) -> List[str]:
    """Greedy decode CTC outputs."""
    pred_ids = torch.argmax(log_probs, dim=-1)  # (batch, time)
    transcripts: List[str] = []
    for seq in pred_ids:
        prev = blank_id
        tokens = []
        for idx in seq.tolist():
            if idx == blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            tokens.append(idx_to_char[idx])
            prev = idx
        transcripts.append("".join(tokens).strip())
    return transcripts


def compute_wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    return wer(list(refs), list(hyps))


def compute_cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    return cer(list(refs), list(hyps))
