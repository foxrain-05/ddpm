import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ import DiffusionModel
from data import DataSet
from torch.utils.data import DataLoader
import cv2

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = DataSet()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        for i, x in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            print(x.shape)

            loss = model.loss_fn(x)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            x = torch.randn(5, 1, 32, 32).to(device)
            ts = torch.arange(model.t_range -1, 0, -1).to(device)

            for t in ts:
                x = model.sample(x, t)
            
            x = x.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255

            for i in range(5):
                cv2.imwrite(f"out/sample{epoch}_{i}.png", x[i])
        

