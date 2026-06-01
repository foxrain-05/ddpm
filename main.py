import os

import cv2
import torch
from torch.utils.data import DataLoader

from data import DataSet
from model import DiffusionModel

batch_size = 128
epochs = 100
sample_dir = "out"
checkpoint_dir = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

dataset = DataSet()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DiffusionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(epochs):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32).to(device)
        ts = torch.arange(model.t_range - 1, 0, -1).to(device)

        for t in ts:
            x = model.sample(x, t)

        x = ((x.clamp(-1, 1) + 1) / 2)
        x = x.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0

        x = cv2.cvtColor(x[0].astype("uint8"), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{sample_dir}/sample{epoch}.jpg", x)

    model.train()
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        loss = model.loss_fn(x)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{checkpoint_dir}/ddpm_epoch_{epoch}.pth",
    )

