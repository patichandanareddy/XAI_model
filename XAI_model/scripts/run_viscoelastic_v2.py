import os, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ---------------------------------------
# Helpers: robust feature selection
# ---------------------------------------
# Candidate names for visco internal strain and its "prev" counterpart
VISC_KEYS = ["e_visc", "e_visco", "eps_v", "e_v", "visc_strain", "ev"]
VISC_PREV_KEYS = ["prev_e_visc", "prev_e_visco", "prev_eps_v", "prev_e_v", "prev_visc_strain", "prev_ev"]

# Base time-series features that almost all datasets have
BASE_KEYS = ["strain", "prev_strain", "prev_stress", "control"]

# Common material parameter names (optional)
PARAM_KEYS = ["E", "eta", "alpha", "beta", "H", "H_kin", "sig0"]

def _first_present_key(d: dict, candidates):
    """Return first key from candidates that exists in dict d, else None."""
    for k in candidates:
        if k in d:
            return k
    return None

def load_one_npy(path):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    elif isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    return d

def build_feature_list(d, use_params=True):
    """Decide which features to use based on what's actually available in this sample."""
    feats = []
    # Always include ones that exist among base keys
    for k in BASE_KEYS:
        if k in d:
            feats.append(k)

    # Try to include visco strain and its prev if available
    vkey = _first_present_key(d, VISC_KEYS)
    if vkey is not None:
        feats.append(vkey)
    pvkey = _first_present_key(d, VISC_PREV_KEYS)
    if pvkey is not None:
        feats.append(pvkey)

    # Include parameters if requested and present
    if use_params:
        for pk in PARAM_KEYS:
            if pk in d:
                feats.append(pk)

    # Ensure uniqueness and stable order
    seen = set()
    uniq = []
    for k in feats:
        if k not in seen:
            uniq.append(k); seen.add(k)
    return uniq

def build_xy(d, feature_names, target_delta=True):
    """Build X, y from a dict and the decided feature list."""
    N = len(d["strain"])  # assume strain is present
    X_cols = []
    for k in feature_names:
        v = d[k]
        if np.ndim(v) == 0:  # broadcast scalars to length N
            v = np.full((N,), float(v))
        arr = np.asarray(v, dtype=np.float32).reshape(-1, 1)
        X_cols.append(arr)
    X = np.concatenate(X_cols, axis=1).astype(np.float32)

    y = np.asarray(d["stress"], dtype=np.float32).reshape(-1, 1)
    if target_delta and "prev_stress" in d:
        prev = np.asarray(d["prev_stress"], dtype=np.float32).reshape(-1, 1)
        y = y - prev
    return X, y

class ViscoDataset(Dataset):
    def __init__(self, files, use_params=True, target_delta=True, limit=None, standardize=False):
        self.files = files
        self.use_params = use_params
        self.target_delta = target_delta
        self.limit = limit
        self.standardize = standardize

        Xs, Ys = [], []
        feature_union = None
        total = 0

        # First pass: determine a *union* of features that appear (robustness),
        # but only among files we actually load (respecting --limit)
        for f in tqdm(files, desc="[Scan]"):
            d = load_one_npy(f)
            feats = build_feature_list(d, use_params=use_params)
            if feature_union is None:
                feature_union = feats
            else:
                # keep only those that are common across all seen so far
                feature_union = [k for k in feature_union if k in feats]
            total += len(d["strain"])
            if limit is not None and total >= limit:
                break

        if not feature_union:
            raise RuntimeError("Could not determine a common feature set across files.")

        # Second pass: actually load arrays using the agreed feature list
        Xs, Ys = [], []
        total = 0
        for f in tqdm(files, desc="[Load]"):
            d = load_one_npy(f)
            X, y = build_xy(d, feature_union, target_delta=target_delta)
            Xs.append(X); Ys.append(y)
            total += len(X)
            if limit is not None and total >= limit:
                break

        self.feature_names = feature_union
        self.X = np.concatenate(Xs, axis=0)
        self.y = np.concatenate(Ys, axis=0)
        if limit is not None and len(self.X) > limit:
            self.X = self.X[:limit]
            self.y = self.y[:limit]

        if standardize:
            self.mean = self.X.mean(axis=0, keepdims=True)
            self.std  = self.X.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std  = None

        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[128,128,128,128,128]):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).squeeze(1)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
    return tot / len(loader.dataset)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).squeeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            tot += loss.item() * len(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    return tot / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)

def plot_loss(train_losses, val_losses, path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def plot_predictions(y_true, y_pred, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)
    m = min(float(y_true.min()), float(y_pred.min()))
    M = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([m, M], [m, M], "r--")
    plt.xlabel("True stress (or Δstress)")
    plt.ylabel("Pred stress (or Δstress)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def plot_residuals(y_true, y_pred, path):
    res = y_pred - y_true
    plt.figure()
    plt.hist(res, bins=100, alpha=0.7)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    print("[Info] Device:", device, "Threads:", args.threads)

    files = sorted(glob.glob(os.path.join(args.data_dir, "INPUTS", "npy", "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No npy files under {args.data_dir}/INPUTS/npy")

    if args.max_files:
        files = files[:args.max_files]

    n_train = int(0.8 * len(files))
    train_files = files[:n_train]
    val_files   = files[n_train:]
    print(f"[Info] Files: train={len(train_files)}  val={len(val_files)}")

    train_ds = ViscoDataset(train_files, use_params=True, target_delta=True,
                            limit=args.limit, standardize=args.standardize)
    val_ds   = ViscoDataset(val_files, use_params=True, target_delta=True,
                            limit=args.limit, standardize=args.standardize)

    # Save the actual features we ended up using (for explain scripts)
    out_dir = os.path.join("outputs", "viscoelastic_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "used_features.json"), "w") as f:
        json.dump(train_ds.feature_names, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)

    model = MLP(train_ds.X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs+1):
        ltr = train_epoch(model, train_loader, opt, loss_fn, device)
        lval, ypred, ytrue = evaluate(model, val_loader, loss_fn, device)
        train_losses.append(ltr)
        val_losses.append(lval)
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {ltr:.6e}")

    torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))
    plot_loss(train_losses, val_losses, os.path.join(out_dir, "loss_curve.png"))
    plot_predictions(ytrue, ypred, os.path.join(out_dir, "predictions_onefile.png"))
    plot_residuals(ytrue, ypred, os.path.join(out_dir, "residuals.png"))

    metrics = {
        "in_dim": train_ds.X.shape[1],
        "feature_names": train_ds.feature_names,
        "use_params": True,
        "target_delta": True,
        "loss": "huber",
        "train_loss_last": float(train_losses[-1]),
        "val_loss_last": float(val_losses[-1]),
        "r2": float(r2_score(ytrue, ypred))
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[Done] Saved:")
    for k in ["model.pt","loss_curve.png","predictions_onefile.png","residuals.png","summary.json","used_features.json"]:
        print(" ", os.path.join(out_dir, k))

if __name__ == "__main__":
    main()
