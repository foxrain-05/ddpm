import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class DataSet(Dataset):
    def __init__(self, root="data", train=True, download=True):
        dataset = CIFAR10(root=root, train=train, download=download)

        self.data = torch.tensor(dataset.data, dtype=torch.float32)
        self.data = self.data / 127.5 - 1.0
        self.data = self.data.moveaxis(3, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
