import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms

class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST"):
        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )

        self.dataset_len = len(train_dataset.data)

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data)
            data = data[:, :, :, None]
            self.depth = 1
            self.size = 32
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data)
            self.depth = 3
            self.size = 32
        self.input_seq = ((data / 255.0) * 2.0) - 1.0
        self.input_seq = self.input_seq.moveaxis(3, 1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item]

if __name__ == "__main__":
    dataset = DiffSet(train=True, dataset="MNIST")
    mnist_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for i, data in enumerate(mnist_dataloader):
        print(data.shape)
        break

    dataset = DiffSet(train=True, dataset="Fashion")
    fashion_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, data in enumerate(fashion_dataloader):
        print(data.shape)
        break

    dataset = DiffSet(train=True, dataset="CIFAR")
    cifar_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, data in enumerate(cifar_dataloader):
        print(data.shape)
        break