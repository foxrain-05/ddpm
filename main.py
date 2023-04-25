import torch
from torch.utils.data import DataLoader
import cv2

from data import DataSet
from model import DiffusionModel

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataSet()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(100):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32).to(device)
        ts = torch.arange(model.t_range -1, 0, -1).to(device)

        for t in ts:
            x = model.sample(x, t)
        
        x = x.permute(0, 2, 3, 1).clamp(0, 1).detach().cpu().numpy() * 255.0

        cv2.imwrite(f"out/sample{epoch}.jpg", x[0])


    model.train()
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        loss = model.loss_fn(x)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
    

