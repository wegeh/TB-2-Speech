"""Utility helpers: decoding and metrics."""

from __future__ import annotations

from typing import Iterable, List, Dict

import random
import numpy as np
import torch
from jiwer import cer, wer


def set_all_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def greedy_decoder(
    log_probs: torch.Tensor,
    idx_to_char: List[str] | Dict[int, str],
    blank_id: int = 0,
) -> List[str]:
    """
    Greedy decode CTC outputs.
    
    Args:
        log_probs: (batch, time, vocab)
        idx_to_char: List or Dict mapping index to character.
        blank_id: Index of the blank token.
        
    Returns:
        List[str]: Decoded transcripts.
    """
    pred_ids = torch.argmax(log_probs, dim=-1)  # (batch, time)
    transcripts: List[str] = []
    
    # Handle Dict input for idx_to_char
    if isinstance(idx_to_char, list):
        get_char = lambda i: idx_to_char[i]
    else:
        get_char = lambda i: idx_to_char.get(i, "")

    for seq in pred_ids:
        prev = blank_id
        tokens = []
        for idx in seq.tolist():
            if idx == blank_id:
                prev = idx
                continue
            if idx == prev:
                continue
            tokens.append(get_char(idx))
            prev = idx
        transcripts.append("".join(tokens).strip())
    return transcripts


def compute_wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Compute Word Error Rate."""
    return wer(list(refs), list(hyps))


def compute_cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Compute Character Error Rate."""
    return cer(list(refs), list(hyps))
