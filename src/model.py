"""Conformer CTC model definition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ConformerCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 256,
        encoder_layers: int = 12,
        num_heads: int = 4,
        n_mels: int = 80,
        dropout: float = 0.1,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=160,
            n_mels=n_mels,
            pad_mode="constant",
            center=True,
        )
        self.input_linear = nn.Linear(n_mels, encoder_dim)
        self.encoder = torchaudio.models.Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=encoder_dim * 4,
            num_layers=encoder_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
        )
        self.fc = nn.Linear(encoder_dim, vocab_size)

    def _feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        n_fft = self.melspec.n_fft
        hop_length = self.melspec.hop_length or 1
        # MelSpectrogram with center=True pads by n_fft//2 on each side.
        pad = n_fft // 2
        lengths = torch.div(
            waveform_lengths + 2 * pad - n_fft, hop_length, rounding_mode="floor"
        ) + 1
        return lengths.clamp(min=1)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor):
        """
        Args:
            waveforms: (batch, time)
            waveform_lengths: (batch,)
        Returns:
            log_probs: (batch, frames, vocab)
            output_lengths: (batch,)
        """
        # Ensure minimum length for STFT padding to avoid reflect pad errors.
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.shape[-1] < self.n_fft:
            pad_amt = self.n_fft - waveforms.shape[-1]
            waveforms = F.pad(waveforms, (0, pad_amt))

        feats = self.melspec(waveforms)  # (batch, n_mels, frames)
        feats = feats.clamp(min=1e-5).log()
        feats = feats.transpose(1, 2)  # (batch, frames, n_mels)
        feats = self.input_linear(feats)

        # Use feature frame lengths derived from original waveforms for the padding mask.
        feature_lengths = self._feature_lengths(waveform_lengths)
        feature_lengths = feature_lengths.clamp(max=feats.size(1))
        encoded, output_lengths = self.encoder(feats, feature_lengths)
        logits = self.fc(encoded)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, output_lengths
