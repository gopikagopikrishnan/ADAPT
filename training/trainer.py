import time
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


class Trainer():
    def __init__(self, dataset, args, use_amp=True, split = 0.8):
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_paths = args.save
        
        # Model Initilization        
        self.model = FixedUNetBeamformer().to(self.device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # Set up directories and logging
        os.makedirs(os.path.join(self.save_paths, "chkpt/", 'iter_' + str(args.run_no)), exist_ok=True)
        os.makedirs(os.path.join(self.save_paths, "logs/"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_paths, 'logs/iter_' + str(args.run_no)))
        self.chkpt = os.path.join(self.save_paths, "chkpt", 'iter_' + str(args.run_no), "model.pt")
        self.bestpt = os.path.join(self.save_paths, "chkpt", 'iter_' + str(args.run_no), "best.pt")
       
        # Train/Validation split
        train_size = int(len(dataset) * split)
        valid_size = len(dataset) - train_size
        self.train_set, self.valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        
        # DataLoader with configurable number of workers
        self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    def compute_loss(self, outputs, targets):
        total_loss = 0
        for i, task in enumerate(['das', 'fdmas', 'capon']):
            pred = outputs[task]
            target = targets[:, i:i+1]
            mse = self.mse_loss(pred, target)
            l1 = self.l1_loss(pred, target)
            task_loss = 0.8 * mse + 0.2 * l1
            total_loss += task_loss
        return total_loss / 3

    def train_epoch(self):
        
        # Load checkpoint if exists
        if os.path.exists(self.chkpt):
            checkpoint = torch.load(self.chkpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', self.scheduler.state_dict()))
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['loss']
        else:
            self.epoch = 0
            self.best_val_loss = float('inf')
            

        
        self.model.train()
        
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            load_start = time.time()
            input_data = batch['input'].to(self.device)
            gt_output = batch['output'].to(self.device)
            load_time = time.time() - load_start
            print(f"Batch {batch_idx}: Data loading time = {load_time:.3f} s")
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    compute_start = time.time()
                    outputs = self.model(input_data)
                    loss = self.compute_loss(outputs, gt_output)
                    compute_time = time.time() - compute_start
                    print(f"Batch {batch_idx}: Forward pass + loss comp. time = {compute_time:.3f} s")

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_data)
                loss = self.compute_loss(outputs, gt_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                input_data = batch['input'].to(self.device)
                gt_output = batch['output'].to(self.device)
                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_data)
                        loss = self.compute_loss(outputs, gt_output)
                else:
                    outputs = self.model(input_data)
                    loss = self.compute_loss(outputs, gt_output)
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def train(self, epochs):
        global_start = time.time()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            start_time = time.time()
            val_start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            val_time = time.time() - val_start
            self.scheduler.step(val_loss)

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Validation time = {val_time:.2f} s")
            print(f"Epoch {epoch + 1} completed in {elapsed:.2f} s")

            total_elapsed = time.time() - global_start
            avg_epoch_time = total_elapsed / (epoch + 1)
            epochs_left = epochs - epoch - 1
            eta_seconds = avg_epoch_time * epochs_left
            print(f"Estimated time remaining: {eta_seconds:.2f} s")
                
            if val_loss < self.best_val_loss:
                print("Saving best model weights...")
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.bestpt)
            
            if epoch % 10 == 0 or val_loss < self.best_val_loss:
                print("Saving model checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': val_loss,
                }, self.chkpt)
                
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            # self.visualize_results(epoch + 1)
        
        self.writer.flush()
        self.writer.close()
        print(f"\nTotal training time: {time.time() - global_start:.2f} s")


    def visualize_results(self, epoch):
        self.model.eval()
        sample = next(iter(self.valid_loader))
        input_data = sample['input'][:1].to(self.device)
        gt_output = sample['output'][:1].to(self.device)
        outputs = self.model(input_data)
        fig, axes = plt.subplots(2, 3, figsize=(4, 12))
        fig.suptitle(f'Epoch {epoch} - GT vs Pred')
        hilbert = HilbertTransform(axis=1)
        for i, task in enumerate(['das', 'fdmas', 'capon']):
            gt_img = gt_output[0, i].cpu()
            gt_env = torch.abs(hilbert(gt_img))
            gt_log = 20 * torch.log10(gt_env / torch.clamp(torch.max(gt_env), min=1e-8))
            gt_norm = (gt_log - torch.min(gt_log)) / (torch.max(gt_log) - torch.min(gt_log) + 1e-8)
            axes[0, i].imshow(gt_norm.numpy().T, cmap='gray', aspect='auto')
            axes[0, i].set_title(f'GT - {task.upper()}')
            axes[0, i].axis('off')
            pred_img = outputs[task][0, 0].cpu()
            pred_env = torch.abs(hilbert(pred_img))
            pred_log = 20 * torch.log10(pred_env / torch.clamp(torch.max(pred_env), min=1e-8))
            pred_norm = (pred_log - torch.min(pred_log)) / (torch.max(pred_log) - torch.min(pred_log) + 1e-8)
            axes[1, i].imshow(pred_norm.numpy().T, cmap='gray', aspect='auto')
            axes[1, i].set_title(f'Pred - {task.upper()}')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(f'/content/results_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def close_writer(self):
        if hasattr(self, 'writer'):
            self.writer.close()
