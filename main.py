import torch
from torchvision.datasets import MNIST
from model import DiffusionModel
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
    ])

    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.eval()
        with torch.no_grad():
            x = torch.randn(5, 1, 32, 32).to(device)
            ts = torch.arange(model.t_range -1, 0, -1).to(device)

            for t in ts:
                x = model.sample(x, t)
            
            x = x.permute(0, 2, 3, 1).cpu().numpy() * 255.0
            
            x = cv2.cvtColor(x[1], cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"out/sample{epoch}.jpg", x)
                

        model.train()
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()

            loss = model.loss_fn(x)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
        

