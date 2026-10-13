import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dataset import PairDataset
from model import Siamese, contrastive_loss

dev = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    S, Y = [], []
    for a, b, y in dl:
        S.append(model(a.to(dev), b.to(dev)).cpu())
        Y.append(y)
    S, Y = torch.cat(S).numpy(), torch.cat(Y).numpy()
    return roc_auc_score(Y, S)


def main():
    print("device:", dev)
    train_dl = DataLoader(PairDataset("data/pairs_train.csv", train=True),
                          batch_size=32, shuffle=True, num_workers=2)
    val_dl = DataLoader(PairDataset("data/pairs_val.csv"),
                        batch_size=64, num_workers=2)

    model = Siamese().to(dev)
    os.makedirs("checkpoints", exist_ok=True)
    drive_ckpt = "/content/drive/MyDrive/atro_return_agent/checkpoints"
    best = 0

    for epoch in range(6):
        freeze = epoch < 1          # epoch 1: train only the new embedding layer
        for n, p in model.net.named_parameters():
            p.requires_grad = (not freeze) or n.startswith("fc")
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=1e-3 if freeze else 1e-4)

        model.train()
        total = 0
        for a, b, y in tqdm(train_dl, desc=f"epoch {epoch+1}"):
            a, b, y = a.to(dev), b.to(dev), y.to(dev)
            loss = contrastive_loss(model(a, b), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        auc = evaluate(model, val_dl)
        print(f"epoch {epoch+1}  loss {total/len(train_dl):.4f}  val AUC {auc:.4f}")

        if auc > best:
            best = auc
            torch.save(model.state_dict(), "checkpoints/siamese_best.pt")
            if os.path.isdir("/content/drive/MyDrive"):      # only true in Colab
                os.makedirs(drive_ckpt, exist_ok=True)
                torch.save(model.state_dict(), f"{drive_ckpt}/siamese_best.pt")


if __name__ == "__main__":
    main()