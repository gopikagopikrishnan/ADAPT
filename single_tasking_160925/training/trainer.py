from monai.losses import SSIMLoss
from torch.optim import AdamW
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.networks.layers import HilbertTransform
import matplotlib.pyplot as plt
from model.tribeamnet_model import FixedUNetBeamformer
from tqdm import tqdm

class Trainer:
    def __init__(self, dataset, args, split=0.8, use_amp=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FixedUNetBeamformer().to(self.device)

        # Loss functions
        self.ssim_loss = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2, data_range=1.0)
        self.l1_loss = nn.L1Loss()

        # Optimizer & scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)

        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Dataset split
        train_size = int(len(dataset) * split)
        valid_size = len(dataset) - train_size
        self.train_set, self.valid_set = random_split(dataset, [train_size, valid_size])
        self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False)

        # Logging and checkpoints
        os.makedirs(args.save, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(args.save, "logs"))
        self.chkpt = os.path.join(args.save, "model.pt")
        self.bestpt = os.path.join(args.save, "best.pt")

        # Save path to stats file (from dataset)
        self.stats_file = getattr(dataset, "stats_file", None)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            x = batch["input"].to(self.device)
            y = batch["output"].to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                w = self.model(x)
                bf = torch.sum(w * x, dim=1, keepdim=True)
                env = torch.abs(HilbertTransform(axis=1)(bf.squeeze(1)))
                log_env = 20 * torch.log10(env + 1e-8)
                pred = torch.clamp(log_env, -80, 0)
                pred = (pred + 80) / 80.0

                ssim = self.ssim_loss(pred.unsqueeze(1), y.unsqueeze(1))
                l1 = self.l1_loss(pred, y)
                loss = 0.8 * ssim + 0.2 * l1

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                x = batch["input"].to(self.device)
                y = batch["output"].to(self.device)
                w = self.model(x)
                bf = torch.sum(w * x, dim=1, keepdim=True)
                env = torch.abs(HilbertTransform(axis=1)(bf.squeeze(1)))
                log_env = 20 * torch.log10(env + 1e-8)
                pred = torch.clamp(log_env, -80, 0)
                pred = (pred + 80) / 80.0

                ssim = self.ssim_loss(pred.unsqueeze(1), y.unsqueeze(1))
                l1 = self.l1_loss(pred, y)
                loss = 0.8 * ssim + 0.2 * l1
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def train(self, epochs, resume=True):
        # Resume if checkpoint exists and resume=True
        if resume and os.path.exists(self.chkpt):
            checkpoint = torch.load(self.chkpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', self.scheduler.state_dict()))
            self.start = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['loss']
            self.stats_file = checkpoint.get('stats_file', self.stats_file)
            print(f"Loaded checkpoint from epoch {self.start-1}. Continuing training.")
        else:
            self.start = 0
            self.best_val_loss = float('inf')


        for epoch in range(self.start, epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            self.scheduler.step(val_loss)

            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                print('************************************************')
                print(f"Saving best model weights at epoch {epoch}")
                print('************************************************')
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.bestpt)

            # Save checkpoint with stats path
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': val_loss,
                'stats_file': self.stats_file
            }, self.chkpt)

            torch.cuda.empty_cache()
            gc.collect()

        self.writer.flush()
        self.writer.close()
