"""
train/train.py
Entry point for ADAPT training.

Usage:-
Train a single task:
    python train/train.py --task das --epochs 100

Train all three tasks sequentially (default):
    python train/train.py --epochs 100

Checkpoint layout produced:
  <save>/
    chkpt/das/iter_<run_no>/model.pt   (latest)
    chkpt/das/iter_<run_no>/best.pt    (best val)
    chkpt/fdmas/iter_<run_no>/...
    chkpt/capon/iter_<run_no>/...
    logs/das/iter_<run_no>/            (TensorBoard)
    logs/fdmas/iter_<run_no>/...
    logs/capon/iter_<run_no>/...
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from types import SimpleNamespace
import numpy as np

from data.dataset import ADAPTDataset
from train.trainer import Trainer

VALID_TASKS = ("das", "fdmas", "capon")


def main_train(
    task:          str  | None = None,   # None → train all three
    dataset_path:  str         = "/content/bmode_dataset_ADAPT",
    save_dir:      str         = "/content/Results",
    epochs:        int         = 100,
    batch_size:    int         = 4,
    num_workers:   int         = 0,
    run_no:        int         = 1,
    resume:        bool        = True,
    use_amp:       bool        = True,
) -> None:
    """Train ADAPT beamformer models.

    Parameters
    ----------
    task :
        ``'das'``, ``'fdmas'``, ``'capon'``, or ``None`` (all three).
    dataset_path :
        Directory of ADAPT-format HDF5 files from generate_bmode_dataset.py.
    save_dir :
        Root directory for checkpoints and TensorBoard logs.
    epochs :
        Total training epochs per task.
    batch_size :
        Mini-batch size.
    num_workers :
        DataLoader workers (0 = main process, safe for Colab).
    run_no :
        Integer run identifier appended to checkpoint/log directories.
    resume :
        Load latest checkpoint and continue if it exists.
    use_amp :
        Mixed-precision training via torch.cuda.amp.
    """
    # Probe geometry for the L11-5v (128 elements, ±19.105 mm)
    probe_geometry = np.linspace(-0.019105, 0.019105, 128)

    tasks_to_run = [task] if task is not None else list(VALID_TASKS)
    if not all(t in VALID_TASKS for t in tasks_to_run):
        raise ValueError(f"task must be one of {VALID_TASKS}")

    args = SimpleNamespace(
        save        = save_dir,
        bs          = batch_size,
        num_workers = num_workers,
        run_no      = run_no,
    )

    for t in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"  Training ADAPT for task: {t.upper()}")
        print(f"{'='*60}")

        dataset = ADAPTDataset(
            folder_path    = dataset_path,
            task           = t,
            probe_geometry = probe_geometry,
        )
        print(f"  Dataset: {len(dataset)} patches from {len(dataset.files)} files")

        trainer = Trainer(dataset, args, use_amp=use_amp)
        trainer.train(epochs=epochs, resume=resume)

    print("\nAll tasks complete.")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ADAPT beamformer")
    parser.add_argument("--task",    type=str,  default=None,
                        choices=[*VALID_TASKS, "all"],
                        help="Beamformer task to train (default: all)")
    parser.add_argument("--dataset", type=str,
                        default="/content/bmode_dataset_ADAPT",
                        help="Path to ADAPT HDF5 dataset")
    parser.add_argument("--save",    type=str,  default="/content/Results")
    parser.add_argument("--epochs",  type=int,  default=100)
    parser.add_argument("--bs",      type=int,  default=4)
    parser.add_argument("--workers", type=int,  default=0)
    parser.add_argument("--run",     type=int,  default=1)
    parser.add_argument("--no-resume", dest="resume",
                        action="store_false", default=True)
    parser.add_argument("--no-amp",  dest="amp",
                        action="store_false", default=True)

    a = parser.parse_args()
    main_train(
        task         = None if a.task in (None, "all") else a.task,
        dataset_path = a.dataset,
        save_dir     = a.save,
        epochs       = a.epochs,
        batch_size   = a.bs,
        num_workers  = a.workers,
        run_no       = a.run,
        resume       = a.resume,
        use_amp      = a.amp,
    )
