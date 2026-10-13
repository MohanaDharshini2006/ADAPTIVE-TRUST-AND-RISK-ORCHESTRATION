import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Siamese(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = nn.Linear(512, dim)          # embedding layer
        self.net = net                         # ONE network = shared weights

    def embed(self, x):
        return F.normalize(self.net(x), dim=1)

    def forward(self, a, b):
        return F.cosine_similarity(self.embed(a), self.embed(b))   # score in [-1, 1]

def contrastive_loss(sim, y, margin=0.4):
    pos = y * (1 - sim) ** 2                                # same product: push similarity to 1
    neg = (1 - y) * torch.clamp(sim - margin, min=0) ** 2   # different: push below the margin
    return (pos + neg).mean()

   # placed after use so the file reads top-dow