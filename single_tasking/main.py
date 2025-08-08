from dataset.patch_dataset_simplified import FixedCustomDatasetTriBeamNet
from training.trainer import Trainer
import os
import argparse


def main():
    
    # Change working directory    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    full_path = os.getcwd()
    
    dataset_path = os.path.dirname(full_path)+os.sep+'Data'
    
    dataset = FixedCustomDatasetTriBeamNet(
        folder_path=dataset_path,
        patch_rows=128,
        seed=42,
        max_files=1000
    ) #taking all the 1000 .h5 files, change max_files for less GT data
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,default = str(os.path.dirname(full_path)+os.sep+'TrainingData'+os.sep+'Bmode_HDF')) 
    parser.add_argument("--save", type=str,default = str(full_path+os.sep+'Results'))
    parser.add_argument("--epochs", type=int,default=300)
    parser.add_argument("--bs", type=int,default=16)
    parser.add_argument("--lr", type=float,default=1e-4)
    parser.add_argument("--run_no", type=int,default=1)
    parser.add_argument("--num_workers", type=int,default=12)
    args = parser.parse_args()
    
    T = Trainer(dataset, args, use_amp=True, split = 0.8)
    T.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
