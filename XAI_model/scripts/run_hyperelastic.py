import os, glob, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class HyperelasticDataset(Dataset):
    def __init__(self, root, limit=None, seed=42, standardize=True):
        # Find strain/stress .npy created by your converter
        x_candidates = [
            os.path.join(root, "INPUTS", "npy", "strain.npy"),
            *glob.glob(os.path.join(root, "INPUTS", "**", "strain*.npy"), recursive=True),
        ]
        y_candidates = [
            os.path.join(root, "RESULTS", "stress.npy"),
            *glob.glob(os.path.join(root, "RESULTS", "**", "stress*.npy"), recursive=True),
        ]
        x_path = next((p for p in x_candidates if os.path.exists(p)), None)
        y_path = next((p for p in y_candidates if os.path.exists(p)), None)
        if x_path is None or y_path is None:
            raise FileNotFoundError(f"Could not find strain/stress npy under {root}")

        x = np.load(x_path).astype(np.float32).reshape(-1, 1)
        y = np.load(y_path).astype(np.float32).reshape(-1, 1)

        # Optional: random subset to keep things fast
        n = min(len(x), len(y))
        idx = np.arange(n)
        if limit is not None and limit < n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=limit, replace=False)
            idx.sort()  # stable order (not required)
        x, y = x[idx], y[idx]

        # Optional: standardize (z-score) for stable training
        self.standardize = standardize
        if standardize:
            self.x_mean, self.x_std = x.mean(), x.std() + 1e-8
            self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
        else:
            self.x_mean = self.x_std = self.y_mean = self.y_std = None

        self.x, self.y = x, y

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, input_size=1, hidden=[112,112,112,112,112], out=1):
        super().__init__()
        layers=[]; d=input_size
        for h in hidden: layers += [nn.Linear(d,h), nn.ReLU()]; d=h
        layers += [nn.Linear(d,out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train(); total=0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--limit", type=int, default=200_000, help="max samples to train on (set None for all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads; 0=leave default")
    ap.add_argument("--standardize", action="store_true", help="z-score inputs/targets (recommended)")
    ap.add_argument("--out_dir", default="outputs/hyperelastic")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    os.makedirs(args.out_dir, exist_ok=True)
    ds = HyperelasticDataset(args.data_dir, limit=args.limit, seed=args.seed, standardize=args.standardize)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    losses=[]
    for ep in range(1, args.epochs+1):
        l = train_epoch(model, loader, opt, loss_fn, device)
        losses.append(l)
        if ep==1 or ep%5==0:
            print(f"Epoch {ep}/{args.epochs}  Loss: {l:.6e}")

    # Loss plot
    plt.figure(); plt.plot(losses); plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("MSE (standardized)" if args.standardize else "MSE")
    plt.title("Hyperelastic: Loss")
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=160); plt.close()

    # Prediction plot (subsample for speed/clarity)
    model.eval()
    x_np, y_np = ds.x, ds.y
    with torch.no_grad():
        p_np = model(torch.from_numpy(x_np)).cpu().numpy()

    # De-standardize to original units for plotting if needed
    if args.standardize:
        p_plot = p_np * ds.y_std + ds.y_mean
        x_plot = x_np * ds.x_std + ds.x_mean
        y_plot = y_np * ds.y_std + ds.y_mean
    else:
        p_plot, x_plot, y_plot = p_np, x_np, y_np

    # Subsample up to 50k for plotting
    max_plot = 50_000
    n = x_plot.shape[0]
    idx = np.arange(n) if n <= max_plot else np.random.choice(n, size=max_plot, replace=False)

    xs = x_plot.squeeze()[idx]
    ys = y_plot.squeeze()[idx]
    ps = p_plot.squeeze()[idx]
    order = np.argsort(xs)
    xs, ys, ps = xs[order], ys[order], ps[order]

    plt.figure()
    plt.plot(xs, ys, label="True")
    plt.plot(xs, ps, "--", label="Pred")
    plt.xlabel("strain"); plt.ylabel("stress"); plt.legend()
    plt.title("Hyperelastic: True vs Pred (subsampled)")
    plt.savefig(os.path.join(args.out_dir, "predictions.png"), dpi=160); plt.close()

if __name__ == "__main__":
    main()
