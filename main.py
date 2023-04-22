import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ import DiffusionModel
from data import DataSet
from torch.utils.data import DataLoader
import cv2

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = DataSet()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.eval()
        with torch.no_grad():
            x = torch.randn(5, 3, 32, 32).to(device)
            ts = torch.arange(model.t_range -1, 0, -1).to(device)

            for t in ts:
                x = model.sample(x, t)
            
            x = x.permute(0, 2, 3, 1).cpu().numpy() * 255

            cv2.imwrite(f"out/sample{epoch}.png", x[1])
                

        model.train()
        for i, x in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()

            loss = model.loss_fn(x)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
        

