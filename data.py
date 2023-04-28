import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

class DataSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.data = CIFAR10(root="data", train=True, transform=self.transform)
        self.data = torch.Tensor(self.data.data)
        self.data = self.data / 255.0
        self.data = self.data.moveaxis(3, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
