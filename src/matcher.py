import torch
from PIL import Image
from model import Siamese
from dataset import eval_tf


def pad_resize(img, size=224):
    img = img.convert("RGB")
    s = max(img.size)
    bg = Image.new("RGB", (s, s), (255, 255, 255))
    bg.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return bg.resize((size, size), Image.BICUBIC)


class Matcher:
    def __init__(self, ckpt="checkpoints/siamese_best.pt"):
        self.model = Siamese()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def score(self, path_a, path_b):
        a = eval_tf(pad_resize(Image.open(path_a)))
        b = eval_tf(pad_resize(Image.open(path_b)))
        return float(self.model(a[None], b[None]))