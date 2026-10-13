import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from dataset import PairDataset
from model import Siamese

dev = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def scores(model, csv):
    dl = DataLoader(PairDataset(csv), batch_size=64, num_workers=2)
    S, Y = [], []
    for a, b, y in dl:
        S.append(model(a.to(dev), b.to(dev)).cpu()); Y.append(y)
    return torch.cat(S).numpy(), torch.cat(Y).numpy()

def main():
    model = Siamese().to(dev)
    model.load_state_dict(torch.load("checkpoints/siamese_best.pt", map_location=dev))
    model.eval()

    vs, vy = scores(model, "data/pairs_val.csv")
    ths = np.linspace(-0.2, 0.95, 116)
    f1 = [precision_recall_fscore_support(vy, vs > t, average="binary", zero_division=0)[2] for t in ths]
    t_star = ths[int(np.argmax(f1))]
    print("Threshold chosen on VALIDATION:", round(float(t_star), 3))

    ts, ty = scores(model, "data/pairs_test.csv")
    p, r, f, _ = precision_recall_fscore_support(ty, ts > t_star, average="binary")
    print(f"TEST  AUC {roc_auc_score(ty, ts):.4f}  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    print(confusion_matrix(ty, ts > t_star))

if __name__ == "__main__":
    main()