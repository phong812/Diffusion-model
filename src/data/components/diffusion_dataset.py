from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
import torch

class DiffusionDataset(Dataset):
  def __init__(self, train=True, dataset_name="FashionMNIST"):
    datasets = {
        "MNIST": MNIST,
        "FashionMNIST": FashionMNIST,
        "CIFAR10": CIFAR10
    }

    # load dataset and transform it to torch.tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets[dataset_name](
        root="./data",
        download=True,
        train=train,
        transform=transform
    )

    # transform dataset
    if dataset_name in ["MNIST", "FashionMNIST"]:
      data = transforms.Pad(2)(dataset.data)  # padding 2 pixels for each border # (60000 x 28 x 28) -> (60000 x 32 x 32)
      data = data.unsqueeze(3)                # add a new dimension # (60000 x 32 x 32) -> (60000 x 32 x 32 x 1)
      self.depth = 1                          # number of channels
      self.size = 32                          # height / width of pictures

    elif dataset_name == "CIFAR10":
      data = torch.Tensor(dataset.data)       # Note: type(cifar10_dataset.data) == numpy.ndarray
      self.depth = 3
      self.size = 32

    # normalise
    self.input_seq = ((data / 255.0) * 2.0) - 1.0   # normalise to [-1, 1]
    self.input_seq = self.input_seq.moveaxis(3, 1)  # (60000 x 32 x 32 x 1) -> (60000 x 1 x 32 x 32)

  def __len__(self):
    return len(self.input_seq)

  def __getitem__(self, index):
    return self.input_seq[index]