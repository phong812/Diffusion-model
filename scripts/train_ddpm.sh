#!/bin/sh
python -m src.train trainer=gpu data.dataset_name="FashionMNIST" logger.wandb.group="ddpm" logger.wandb.name="FashionMNIST"