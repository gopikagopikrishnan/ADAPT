from dataset.patch_dataset import FixedCustomDatasetTriBeamNet
from training.trainer import FixedTrainer

def main():
    dataset_path = "/content/content/beamformed_dataset"
    dataset = FixedCustomDatasetTriBeamNet(
        folder_path=dataset_path,
        patch_rows=128,
        seed=42,
        max_files=1000
    )
#taking all the 1000 .h5 files, change max_files for less GT data
    trainer = FixedTrainer(
        dataset=dataset,
        batch_size=4,
        lr=1e-4,
        num_workers=2,
        use_amp=True
    )

    trainer.train(epochs=20)
    trainer.close_writer()

if __name__ == "__main__":
    main()
