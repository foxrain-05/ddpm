import torch

a = torch.asarray([1, 2, 3])
b = torch.asarray([1, 2, 1])

print(torch.gather(a, 0, b))