import torch

aspects = torch.randn(100, 20).unsqueeze(0)
print(aspects.size())
b = torch.randn(1, 200, 20)
a = torch.cat([aspects, b], dim=1)
print(a.size())