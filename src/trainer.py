"""Training utilities for Conformer CTC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.utils import compute_cer, compute_wer, greedy_decoder


class CTCTrainer:
    """
    Trainer for CTC-based models.
    
    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        device (torch.device): Device to train on.
        idx_to_char (List[str] | Dict[int, str]): Mapping from index to character.
        blank_id (int): Index of the blank token.
        grad_clip (float): Gradient clipping value.
        log_interval (int): Steps between logging.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        idx_to_char: List[str] | Dict[int, str],
        blank_id: int = 0,
        grad_clip: float = 5.0,
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.idx_to_char = idx_to_char
        self.blank_id = blank_id
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def _step_batch(self, batch: Dict, train: bool = True) -> float:
        """
        Process a single batch.
        
        Args:
            batch (Dict): Batch dictionary containing waveforms, labels, etc.
            train (bool): Whether to perform backpropagation.
            
        Returns:
            float: Loss value.
        """
        waveforms = batch["waveforms"].to(self.device)
        waveform_lengths = batch["waveform_lengths"].to(self.device)
        labels = batch["labels"].to(self.device)
        label_lengths = batch["label_lengths"].to(self.device)

        if train:
            self.optimizer.zero_grad()

        log_probs, output_lengths = self.model(waveforms, waveform_lengths)
        loss = self.ctc_loss(log_probs.permute(1, 0, 2), labels, output_lengths, label_lengths)

        if train:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader (DataLoader): Training dataloader.
            epoch (int): Current epoch number.
            
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            loss = self._step_batch(batch, train=True)
            total_loss += loss
            if step % self.log_interval == 0:
                print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | Loss {loss:.4f}")
        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader (DataLoader): Validation dataloader.
            
        Returns:
            Dict[str, float]: Dictionary with 'loss', 'wer', 'cer'.
        """
        self.model.eval()
        total_loss = 0.0
        refs = []
        hyps = []

        with torch.no_grad():
            for batch in dataloader:
                waveforms = batch["waveforms"].to(self.device)
                waveform_lengths = batch["waveform_lengths"].to(self.device)
                labels = batch["labels"].to(self.device)
                label_lengths = batch["label_lengths"].to(self.device)

                log_probs, output_lengths = self.model(waveforms, waveform_lengths)
                loss = self.ctc_loss(log_probs.permute(1, 0, 2), labels, output_lengths, label_lengths)
                total_loss += loss.item()

                decoded = greedy_decoder(log_probs, self.idx_to_char, blank_id=self.blank_id)

                refs.extend(batch["transcripts"])
                hyps.extend(decoded)

        avg_loss = total_loss / max(1, len(dataloader))
        wer_score = compute_wer(refs, hyps) if refs else 0.0
        cer_score = compute_cer(refs, hyps) if refs else 0.0
        return {"loss": avg_loss, "wer": wer_score, "cer": cer_score}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Path,
    ):
        """
        Run the training loop.
        
        Args:
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            epochs (int): Number of epochs to train.
            checkpoint_dir (Path): Directory to save checkpoints.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_wer = float("inf")
        best_path = checkpoint_dir / "best_model.pth"

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            print(
                f"Epoch {epoch} | Train Loss {train_loss:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"Val WER {val_metrics['wer']:.4f} | Val CER {val_metrics['cer']:.4f}"
            )

            if val_metrics["wer"] < best_wer:
                best_wer = val_metrics["wer"]
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "wer": best_wer,
                    },
                    best_path,
                )
                print(f"Saved new best model to {best_path}")
