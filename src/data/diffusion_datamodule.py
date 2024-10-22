from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.diffusion_dataset import DiffusionDataset

class DiffusionDataModule(LightningDataModule):
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

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = DiffusionDataset(train=True, dataset_name=self.hparams.dataset_name)
            testset = DiffusionDataset(train=False, dataset_name=self.hparams.dataset_name)
            dataset = ConcatDataset(datasets=[trainset, testset])
            
            self.data_train, self.data_val, self.data_test = random_split(
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


############################################################### TEST ###############################################################
import hydra
from omegaconf import DictConfig
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # test instance
    dm: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    
    # test dataloader
    print(f"Length of train dataloader: {len(dm.train_dataloader())}")
    print(f"Length of val dataloader: {len(dm.val_dataloader())}")
    print(f"Length of test dataloader: {len(dm.test_dataloader())}\n")
    
    # test batch
    batch = next(iter(dm.train_dataloader()))
    print(f"Type of one batch: {type(batch)}")
    print(f"Length of one batch (batch size): {len(batch)}")
    print(f"Shape of one sample: {batch[0].size()}")

if __name__ == "__main__":
    main()
############################################################### TEST ###############################################################