from dataset.patch_dataset_simplified import FixedCustomDatasetTriBeamNet
from training.trainer import Trainer
import os
import argparse
import numpy as np

def main():
    # Ensure working directory is the script's location
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    full_path = os.getcwd()

    # Default dataset path
    dataset_path = os.path.join(os.path.dirname(full_path), "Data")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Ultrasound Capon Beamformer Training")
    parser.add_argument("--data", type=str, default=dataset_path,
                        help="Path to dataset directory containing .h5 files")
    parser.add_argument("--save", type=str, default=os.path.join(full_path, "Results"),
                        help="Path to save models, logs, and stats")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs")
    parser.add_argument("--bs", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--run_no", type=int, default=1,
                        help="Run identifier")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of DataLoader workers")
    parser.add_argument("--patch_rows", type=int, default=128,
                        help="Patch height in rows")
    parser.add_argument("--max_files", type=int, default=1000,
                        help="Max number of .h5 files to load")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fs", type=int, default=31250000,
                        help="Sampling frequency")
    args = parser.parse_args()

    # Initialize dataset
    dataset = FixedCustomDatasetTriBeamNet(
        folder_path=args.data,
        patch_rows=args.patch_rows,
        seed=args.seed,
        max_files=args.max_files,
        fs=args.fs,
        save_stats_path=os.path.join(args.save, "global_stats.npz")
    )

    # Initialize trainer
    trainer = Trainer(dataset, args, use_amp=True, split=0.8)

    # Train
    trainer.train(epochs=args.epochs)

    print("Training complete. Best model and normalization stats saved.")

if __name__ == "__main__":
    main()
