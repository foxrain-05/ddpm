import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from glob import glob
import numpy as np

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
    

class ImageSet(Dataset):
    def __init__(self):
        self.data = glob("np_images/*.npy")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        image = np.load(self.data[item])
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    dataset = DataSet()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for i, x in enumerate(dataloader):
        print(x.shape)
        
        break
