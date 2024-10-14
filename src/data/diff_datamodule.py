from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from omegaconf import DictConfig, OmegaConf
import hydra
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import rootutils


path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.diff_dataset import DiffusionDataset


class DIFF_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (60_000, 10_000, 0),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = "FashionMNIST",
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size


    def setup(self, stage: Optional[str] = None) -> None:
        
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = DiffusionDataset(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = DiffusionDataset(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            
            self.data_train, self.data_val, _ = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
def print_min_max(batch_tensor):
    print(f"min_value: {torch.min(batch_tensor).item()}, max value: {torch.max(batch_tensor).item()}")
    
def imshow(img):
    batch_size = img.size(0)
    plt.figure(figsize=(8, 8))
    for i in range(batch_size):
        plt.subplot(4, 8, i + 1)
        plt.imshow(img[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()



config_path = str(path / "configs" / "data")

@hydra.main(version_base = "1.3", config_path=config_path, config_name="diff_dataset.yaml")       
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dm: LightningDataModule = hydra.utils.instantiate(cfg)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    bx = next(iter(train_loader))
    print(f"type bx: {type(bx)}")
    print_min_max(bx)
        
    imshow(bx)
        
    print("n_batches", len(train_loader), bx.shape)
        
    for bx in tqdm(train_loader):
        pass
    print("training data passed")
      
    for bx in tqdm(val_loader):
        pass
    print("validation data passed")
                
                
if __name__ == "__main__":  
    main()
