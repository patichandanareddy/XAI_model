# scripts/run_elastoplastic.py
import os
import glob
import json
import argparse
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------
# Dataset
# ---------------------------
class ElastoPlastDataset(Dataset):
    """
    Loads dict-like .npy samples produced for the elastoplasticity task.
    Expected keys per file (example): 'strain', 'stress', 'e_plastic', 'control', 'time', ...
    We learn mapping: X = [strain, prev_strain, prev_stress, prev_e_plastic, control, time] -> Y = stress.
    If some optional keys are missing, we degrade gracefully using zeros.
    """
    def __init__(
        self,
        files: List[str],
        standardize: bool = False,
        limit: Optional[int] = None,
        stats: Optional[dict] = None,
    ):
        self.files = files
        if limit is not None:
            # Use only first 'limit' rows (across files)
            self.limit_rows = int(limit)
        else:
            self.limit_rows = None

        self.standardize = standardize
        self.X, self.Y = self._load_all()
        # compute/assign stats if needed
        if self.standardize:
            if stats is None:
                self.x_mean = self.X.mean(axis=0, keepdims=True)
                self.x_std = self.X.std(axis=0, keepdims=True) + 1e-8
                self.y_mean = self.Y.mean(axis=0, keepdims=True)
                self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-8
            else:
                self.x_mean = stats["x_mean"]
                self.x_std = stats["x_std"]
                self.y_mean = stats["y_mean"]
                self.y_std = stats["y_std"]

            self.X = (self.X - self.x_mean) / self.x_std
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None

    def _safe_get(self, d: dict, key: str, fallback: Optional[np.ndarray] = None):
        if key in d and isinstance(d[key], np.ndarray):
            return d[key]
        return fallback

    def _load_one(self, fpath: str) -> Tuple[np.ndarray, np.ndarray]:
        d = np.load(fpath, allow_pickle=True).item()

        # Required targets
        stress = self._safe_get(d, "stress")
        if stress is None:
            raise ValueError(f"'stress' missing in {fpath}")

        # Inputs (fill missing with zeros of matching length)
        strain = self._safe_get(d, "strain")
        n = stress.shape[0]
        def z(): return np.zeros(n, dtype=np.float64)

        prev_strain = self._safe_get(d, "prev_strain", z())
        prev_stress = self._safe_get(d, "prev_stress", z())
        prev_e_plastic = self._safe_get(d, "prev_e_plastic", z())
        control = self._safe_get(d, "control", z())
        time = self._safe_get(d, "time", z())

        if strain is None:
            raise ValueError(f"'strain' missing in {fpath}")

        # Truncate to same length (defensive)
        m = min(n, strain.shape[0], prev_strain.shape[0], prev_stress.shape[0],
                prev_e_plastic.shape[0], control.shape[0], time.shape[0])

        x = np.stack([
            strain[:m],
            prev_strain[:m],
            prev_stress[:m],
            prev_e_plastic[:m],
            control[:m],
            time[:m]
        ], axis=1).astype(np.float32)

        y = stress[:m].astype(np.float32).reshape(-1, 1)
        return x, y

    def _load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        Xs, Ys = [], []
        rows_accum = 0
        for f in self.files:
            x, y = self._load_one(f)
            if self.limit_rows is not None:
                remaining = self.limit_rows - rows_accum
                if remaining <= 0:
                    break
                if x.shape[0] > remaining:
                    x = x[:remaining]
                    y = y[:remaining]
            Xs.append(x)
            Ys.append(y)
            rows_accum += x.shape[0]
        X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 6), dtype=np.float32)
        Y = np.concatenate(Ys, axis=0) if Ys else np.zeros((0, 1), dtype=np.float32)
        print(f"[Dataset] Train samples: {X.shape[0]}  Val samples: {Y.shape[0]}" if len(self.files) else "")
        return X, Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return numpy for speed; DataLoader collate will convert to tensor
        return self.X[idx], self.Y[idx]


def np_collate(batch):
    X = np.stack([b[0] for b in batch], axis=0)
    Y = np.stack([b[1] for b in batch], axis=0)
    return torch.from_numpy(X), torch.from_numpy(Y)


# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=6, hidden=128, depth=3, out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Training / Eval
# ---------------------------
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        bs = xb.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = xb.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


# ---------------------------
# Plotting
# ---------------------------
def plot_loss(train_hist, val_hist, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def plot_predictions(model, ds: ElastoPlastDataset, out_path: str, device, n_series: int = 3):
    """
    Take a few contiguous chunks from validation set and plot true vs predicted stress.
    Works whether ds tensors live on CPU or are numpy.
    """
    model.eval()
    N = len(ds)
    if N == 0:
        return
    idx0 = np.linspace(0, max(0, N-1000), num=n_series, dtype=int)

    plt.figure(figsize=(8, 6))
    for i, start in enumerate(idx0):
        end = min(N, start + 1000)
        X = ds.X[start:end]
        Y = ds.Y[start:end]

        # to torch
        x_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        y_pred = model(x_t).cpu().numpy().reshape(-1)

        y_true = Y.reshape(-1)

        # un-standardize back to original scale if needed
        if ds.standardize and ds.y_mean is not None:
            y_pred = y_pred * ds.y_std.squeeze() + ds.y_mean.squeeze()
            y_true = y_true * ds.y_std.squeeze() + ds.y_mean.squeeze()

        t = np.arange(start, end)
        plt.plot(t, y_true, linewidth=1.0, alpha=0.8, label=f"true_{i+1}" if i==0 else None)
        plt.plot(t, y_pred, linewidth=1.0, alpha=0.8, linestyle="--", label=f"pred_{i+1}" if i==0 else None)

    plt.xlabel("Sample index")
    plt.ylabel("Stress")
    plt.title("Elastoplastic: True vs Predicted")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------
# Helpers
# ---------------------------
def split_files(all_files: List[str], n_val: int = 1000) -> Tuple[List[str], List[str]]:
    all_files = sorted(all_files)
    if len(all_files) <= n_val:
        return [], all_files
    # put last n_val as validation to avoid overlap if sorted has structure
    train_files = all_files[:-n_val]
    val_files = all_files[-n_val:]
    return train_files, val_files


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    for pg in optimizer.param_groups:
        return float(pg.get("lr", 0.0))
    return 0.0


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="root dir e.g. data/elastoplasticity")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads and DataLoader workers")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--max_files", type=int, default=None, help="limit number of training files")
    ap.add_argument("--limit", type=int, default=None, help="limit total samples used from dataset")
    # model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    # scheduler
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_patience", type=int, default=3)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)
    # early stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=10)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    print(f"[Info] Device: {device.type}, threads: {args.threads or 'default'}")

    # find files
    npy_dir = os.path.join(args.data_dir, "INPUTS", "npy")
    files = sorted(glob.glob(os.path.join(npy_dir, "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No files like {npy_dir}/sample_*.npy")

    # split train/val
    train_files, val_files = split_files(files, n_val=1000)
    if args.max_files is not None and args.max_files > 0:
        train_files = train_files[:args.max_files]
    print(f"[Info] Files: train={len(train_files)}  val={len(val_files)}")

    # build a small temporary dataset to compute standardization stats (on train only)
    tmp_train = ElastoPlastDataset(train_files, standardize=False, limit=args.limit)
    if len(tmp_train) == 0:
        raise RuntimeError("No training samples found.")
    stats = None
    if args.standardize:
        stats = {
            "x_mean": tmp_train.X.mean(axis=0, keepdims=True),
            "x_std": tmp_train.X.std(axis=0, keepdims=True) + 1e-8,
            "y_mean": tmp_train.Y.mean(axis=0, keepdims=True),
            "y_std": tmp_train.Y.std(axis=0, keepdims=True) + 1e-8,
        }

    # actual datasets (with/without standardization)
    train_ds = ElastoPlastDataset(train_files, standardize=args.standardize, limit=args.limit, stats=stats)
    val_ds = ElastoPlastDataset(val_files, standardize=args.standardize, limit=args.limit, stats=stats)

    # loaders
    num_workers = max(0, min(args.threads or 0, 4))  # safe on Windows
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              collate_fn=np_collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=np_collate, drop_last=False)

    # model/opt/loss
    model = MLP(in_dim=train_ds.X.shape[1], hidden=args.hidden, depth=args.depth,
                out_dim=1, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # scheduler (no 'verbose' arg used)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.sched_factor,
        patience=args.sched_patience, min_lr=args.sched_min_lr
    )

    # training loop
    out_dir = os.path.join("outputs", "elastoplastic")
    os.makedirs(out_dir, exist_ok=True)
    train_hist, val_hist = [], []

    best_val = math.inf
    best_epoch = -1
    epochs_no_improve = 0

    print(f"[Dataset] Loaded {len(train_ds)} train samples, {len(val_ds)} val samples")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        va_loss = eval_epoch(model, val_loader, loss_fn, device)
        train_hist.append(tr_loss)
        val_hist.append(va_loss)

        # step scheduler on validation loss, report LR if changed
        prev_lr = current_lr(opt)
        scheduler.step(va_loss)
        new_lr = current_lr(opt)
        lr_msg = f"  [LR {prev_lr:.2e}->{new_lr:.2e}]" if abs(new_lr - prev_lr) > 1e-14 else ""

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {tr_loss:.9e}  ValLoss: {va_loss:.9e}{lr_msg}")

        # early stopping
        if args.early_stop:
            if va_loss + 1e-12 < best_val:
                best_val = va_loss
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.early_stop_patience:
                    print(f"[EarlyStop] No improvement for {args.early_stop_patience} epochs. "
                          f"Best Val: {best_val:.6e} at epoch {best_epoch}.")
                    break

    # save artifacts
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    plot_loss(train_hist, val_hist, os.path.join(out_dir, "loss_curve.png"))
    plot_predictions(model, val_ds, os.path.join(out_dir, "predictions.png"), device)

    # write a small summary
    summary = {
        "train_loss_last": train_hist[-1] if train_hist else None,
        "val_loss_last": val_hist[-1] if val_hist else None,
        "best_val_loss": float(min(val_hist)) if val_hist else None,
        "epochs_ran": len(train_hist)
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Saved:")
    print(f"  {os.path.join(out_dir, 'model.pt')}")
    print(f"  {os.path.join(out_dir, 'loss_curve.png')}")
    print(f"  {os.path.join(out_dir, 'predictions.png')}")


if __name__ == "__main__":
    main()
