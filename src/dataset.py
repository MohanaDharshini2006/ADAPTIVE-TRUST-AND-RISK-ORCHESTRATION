import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])

class PairDataset(Dataset):
    def __init__(self, csv_path, crops_dir="data/crops", train=False):
        self.pairs = pd.read_csv(csv_path)
        self.crops_dir = crops_dir
        self.tf = train_tf if train else eval_tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        r = self.pairs.iloc[i]
        a = Image.open(f"{self.crops_dir}/{r.img_a}.jpg").convert("RGB")
        b = Image.open(f"{self.crops_dir}/{r.img_b}.jpg").convert("RGB")
        return self.tf(a), self.tf(b), torch.tensor(r.label, dtype=torch.float32)