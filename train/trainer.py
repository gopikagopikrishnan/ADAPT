"""
train/trainer.py
Task-specific Trainer for a single FixedUNetBeamformer.

Training signal
───────────────
  1. Apply softmax weights w to ToFC input x:
       bf   = Σ_elements (w · x) = weighted beamforming
  2. Envelope detection via Hilbert transform along depth axis.
  3. Log-compression → normalise to [0, 1] using [-60, 0] dB range.
  4. SSIM loss against GT b-mode (also in [0, 1]).

Checkpoint layout:-
  <save_dir>/
    chkpt/iter_<run_no>/model.pt   — latest checkpoint (resume-safe)
    chkpt/iter_<run_no>/best.pt    — best validation weights only
    logs/iter_<run_no>/            — TensorBoard event files
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gc
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import monai.losses
from monai.networks.layers import HilbertTransform

from model.adapt_model import FixedUNetBeamformer
from configs.config import (
    LR, WEIGHT_DECAY, SCHEDULER_PAT, SCHEDULER_FACTOR, GRAD_CLIP,
    SPLIT, SEED, DB_MIN, DB_MAX,
)


class Trainer:
    """Train one FixedUNetBeamformer for a single beamforming task.

    Parameters
    dataset :
        ADAPTDataset initialised with the desired task.
    args :
        SimpleNamespace with fields:
          save        - root directory for checkpoints and logs
          bs          - batch size
          num_workers - DataLoader workers
          run_no      - run identifier (used in directory names)
    use_amp :
        Enable automatic mixed precision (recommended).
    split :
        Train / validation split ratio.
    """

    def __init__(
        self,
        dataset,
        args:    SimpleNamespace,
        use_amp: bool  = True,
        split:   float = SPLIT,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trainer] device={self.device}  task={dataset.task}")

        self.task    = dataset.task
        self.use_amp = use_amp

        # Model
        self.model = FixedUNetBeamformer().to(self.device)

        # Loss
        self.ssim_loss = monai.losses.SSIMLoss(spatial_dims=2, data_range=1.0)

        # Optimiser & scheduler
        self.optimizer = AdamW(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=SCHEDULER_PAT, factor=SCHEDULER_FACTOR,
        )
        self.scaler = GradScaler() if use_amp else None

        # Data split
        train_size = int(len(dataset) * split)
        valid_size = len(dataset) - train_size
        self.train_set, self.valid_set = random_split(
            dataset, [train_size, valid_size],
            generator=torch.Generator().manual_seed(SEED),
        )
        kw = dict(batch_size=args.bs, num_workers=args.num_workers, pin_memory=True)
        self.train_loader = DataLoader(self.train_set, shuffle=True,  **kw)
        self.valid_loader = DataLoader(self.valid_set, shuffle=False, **kw)

        # Directories & logging
        tag = f"{self.task}/iter_{args.run_no}"
        chkpt_dir = os.path.join(args.save, "chkpt", tag)
        os.makedirs(chkpt_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save, "logs"), exist_ok=True)

        self.writer  = SummaryWriter(os.path.join(args.save, "logs", tag))
        self.chkpt   = os.path.join(chkpt_dir, "model.pt")
        self.bestpt  = os.path.join(chkpt_dir, "best.pt")

    # Beamforming + normalisation

    @staticmethod
    def _bmode_prediction(
        weights: torch.Tensor,   # (B, N_ELEM, H, W)
        x:       torch.Tensor,   # (B, N_ELEM, H, W)
    ) -> torch.Tensor:
        """Apply apodization weights, Hilbert, log-compress → [0, 1].

        Returns (B, H, W).
        """
        bf      = torch.sum(weights * x, dim=1)           # (B, H, W)
        hilbert = HilbertTransform(axis=1)                 # Hilbert along depth
        env     = torch.abs(hilbert(bf))                   # (B, H, W)
        log_env = 20.0 * torch.log10(env + 1e-8)
        pred    = torch.clamp(log_env, DB_MIN, DB_MAX)
        pred    = (pred - DB_MIN) / (DB_MAX - DB_MIN)      # → [0, 1]
        return pred

    # Single epoch

    def _forward(
        self,
        batch: dict,
    ) -> torch.Tensor:
        x = batch["input"].to(self.device)           # (B, 128, H, W)
        y = batch["output"].to(self.device)          # (B, H, W)
        w = self.model(x)                            # (B, 128, H, W)
        pred = self._bmode_prediction(w, x)          # (B, H, W)
        # SSIM expects (B, C, H, W)
        loss = self.ssim_loss(pred.unsqueeze(1), y.unsqueeze(1))
        return loss

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    loss = self._forward(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._forward(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

            total += loss.item()

        return total / len(self.train_loader)

    def validate_epoch(self) -> float:
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for batch in self.valid_loader:
                if self.use_amp:
                    with autocast():
                        loss = self._forward(batch)
                else:
                    loss = self._forward(batch)
                total += loss.item()

        return total / len(self.valid_loader)

    # Full training loop

    def train(self, epochs: int, resume: bool = True) -> None:
        """Train for ``epochs`` epochs, optionally resuming from checkpoint.

        Parameters
        epochs : total number of epochs.
        resume : if True and a checkpoint exists, continue from last epoch.
        """
        # ── Resume ────────────────────────────────────────────────────────────
        if resume and os.path.exists(self.chkpt):
            ckpt = torch.load(self.chkpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(
                ckpt.get("scheduler_state_dict", self.scheduler.state_dict())
            )
            start          = ckpt["epoch"] + 1
            best_val_loss  = ckpt["loss"]
            print(f"[Trainer/{self.task}] Resumed from epoch {start - 1}.")
        else:
            start         = 0
            best_val_loss = float("inf")

        # Loop
        for epoch in range(start, epochs):
            train_loss = self.train_epoch()
            val_loss   = self.validate_epoch()
            self.scheduler.step(val_loss)

            # Logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val",   val_loss,   epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )
            print(
                f"[{self.task.upper():5s}] epoch {epoch:>4d}/{epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}"
            )

            # Save best weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.bestpt)
                print(f"  ↳ New best val loss ({val_loss:.5f}) — saved best.pt")

            # Save resumable checkpoint
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "loss":                 val_loss,
                },
                self.chkpt,
            )

            torch.cuda.empty_cache()
            gc.collect()

        self.writer.flush()
        self.writer.close()
        print(f"[Trainer/{self.task}] Training complete.  Best val={best_val_loss:.5f}")
