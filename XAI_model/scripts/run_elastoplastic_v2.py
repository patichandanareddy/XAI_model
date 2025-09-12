import os, glob, json, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------
# Utils
# ---------------------------
def set_threads(n):
    if n and n > 0:
        torch.set_num_threads(n)
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std  = x.std(axis=0, keepdims=True) + 1e-8
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean

# ---------------------------
# Dataset
# ---------------------------
class ElastoPlastDataset(Dataset):
    """
    Each file is a dict-like object saved in .npy with keys:
    'strain','stress','e_plastic','control','time','prev_e_plastic','prev_stress','prev_strain','E','sig0','H_kin'
    We flatten per-file series into samples (row-wise).
    """
    def __init__(self, files, standardize=True, limit=None, use_params=False, target_delta=False, fit_scalers=False, x_scaler=None, y_scaler=None):
        self.files = files
        self.use_params = use_params
        self.target_delta = target_delta

        Xs, Ys, aux = [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True).item()
            # Series
            strain = d["strain"].astype(np.float32).reshape(-1,1)
            e_pl   = d["e_plastic"].astype(np.float32).reshape(-1,1)
            ctrl   = d.get("control", np.zeros_like(strain)).astype(np.float32).reshape(-1,1)
            ps     = d["prev_stress"].astype(np.float32).reshape(-1,1)
            pstr   = d["prev_strain"].astype(np.float32).reshape(-1,1)
            pepl   = d["prev_e_plastic"].astype(np.float32).reshape(-1,1)
            y      = d["stress"].astype(np.float32).reshape(-1,1)

            # Optional constants as features (broadcast per time-step)
            add = []
            if self.use_params:
                E     = np.float32(d.get("E", 0.0))
                sig0  = np.float32(d.get("sig0", 0.0))
                Hkin  = np.float32(d.get("H_kin", 0.0))
                const = np.repeat(np.array([[E, sig0, Hkin]], dtype=np.float32), strain.shape[0], axis=0)
                add.append(const)

            X = np.concatenate([strain, e_pl, pstr, ps, pepl, ctrl] + add, axis=1)

            if self.target_delta:
                y = y - ps  # Δσ_t = σ_t - σ_{t-1}

            Xs.append(X)
            Ys.append(y)
            # keep masks for diagnostics
            aux.append({
                "elastic_mask": (e_pl.reshape(-1) == 0.0),
                "prev_stress": ps.reshape(-1),
                "stress_abs": (y + (ps if self.target_delta else 0)).reshape(-1),
                "strain": strain.reshape(-1)
            })

        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)

        if limit is not None:
            n = min(limit, X.shape[0])
            X, Y = X[:n], Y[:n]
            # trim aux arrays too
            merged = {k: np.concatenate([a[k] for a in aux], axis=0)[:n] for k in aux[0].keys()}
        else:
            merged = {k: np.concatenate([a[k] for a in aux], axis=0) for k in aux[0].keys()}

        self.X_raw = X
        self.Y_raw = Y
        self.aux = merged

        # Standardization (fit on train only)
        self.x_scaler = x_scaler or Standardizer()
        self.y_scaler = y_scaler or Standardizer()
        self.standardize = standardize
        if standardize and fit_scalers:
            self.x_scaler.fit(self.X_raw)
            self.y_scaler.fit(self.Y_raw)

        if standardize:
            self.X = self.x_scaler.transform(self.X_raw)
            self.Y = self.y_scaler.transform(self.Y_raw)
        else:
            self.X, self.Y = self.X_raw, self.Y_raw

        self.n = self.X.shape[0]

    def __len__(self): return self.n
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=4, out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        h = in_dim
        for _ in range(depth):
            layers += [nn.Linear(h, hidden), nn.ReLU()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            h = hidden
        layers += [nn.Linear(h, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Training / Eval
# ---------------------------
def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    preds, trues, strains, elastic_mask = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    loss = total / len(loader.dataset)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return loss, preds, trues

# ---------------------------
# Plot helpers
# ---------------------------
def plot_loss(train_hist, val_hist, path):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_predictions_single_file(model, ds_val_fullfile, out_png, device, target_delta, y_scaler, title="Prediction (one file)"):
    """
    For interpretability, plot on one validation FILE (not the shuffled pool).
    We load the first available file and reconstruct stress if target_delta.
    """
    # pick the first file of the validation list
    f = ds_val_fullfile.files[0]
    d = np.load(f, allow_pickle=True).item()
    strain = d["strain"].astype(np.float32).reshape(-1,1)
    e_pl   = d["e_plastic"].astype(np.float32).reshape(-1,1)
    ctrl   = d.get("control", np.zeros_like(strain)).astype(np.float32).reshape(-1,1)
    ps     = d["prev_stress"].astype(np.float32).reshape(-1,1)
    pstr   = d["prev_strain"].astype(np.float32).reshape(-1,1)
    pepl   = d["prev_e_plastic"].astype(np.float32).reshape(-1,1)
    y      = d["stress"].astype(np.float32).reshape(-1,1)

    add = []
    if ds_val_fullfile.use_params:
        E     = np.float32(d.get("E", 0.0))
        sig0  = np.float32(d.get("sig0", 0.0))
        Hkin  = np.float32(d.get("H_kin", 0.0))
        const = np.repeat(np.array([[E, sig0, Hkin]], dtype=np.float32), strain.shape[0], axis=0)
        add.append(const)
    X = np.concatenate([strain, e_pl, pstr, ps, pepl, ctrl] + add, axis=1)
    Xs = ds_val_fullfile.x_scaler.transform(X) if ds_val_fullfile.standardize else X

    with torch.no_grad():
        yp = model(torch.from_numpy(Xs).to(device)).cpu().numpy()

    if ds_val_fullfile.standardize:
        yp = y_scaler.inverse(yp)

    if target_delta:
        # reconstruct stress from prev_stress + cumsum(Δσ)
        sigma_pred = ps + np.cumsum(yp, axis=0)
        sigma_true = y
    else:
        sigma_pred = yp
        sigma_true = y

    plt.figure(figsize=(7,4))
    plt.plot(strain.reshape(-1), sigma_true.reshape(-1), label="true", linewidth=1)
    plt.plot(strain.reshape(-1), sigma_pred.reshape(-1), label="pred", linewidth=1)
    plt.xlabel("strain")
    plt.ylabel("stress")
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_residuals(y_true, y_pred, strain_like, out_png, title="Residuals"):
    res = (y_pred - y_true).reshape(-1)
    plt.figure(figsize=(6,4))
    plt.scatter(strain_like.reshape(-1)[:len(res)], res, s=1)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xlabel("strain (sample-aligned)")
    plt.ylabel("pred - true")
    plt.title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--max_files", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_patience", type=int, default=5)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=12)
    ap.add_argument("--out_dir", default="outputs/elastoplastic")
    ap.add_argument("--use_params", action="store_true", help="append E, sig0, H_kin to features")
    ap.add_argument("--target_delta", action="store_true", help="predict Δstress instead of absolute")
    ap.add_argument("--loss", choices=["huber","mse"], default="huber")
    args = ap.parse_args()

    set_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}, threads: {args.threads}")

    # files
    npy_dir = os.path.join(args.data_dir, "INPUTS", "npy")
    files = sorted(glob.glob(os.path.join(npy_dir, "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No files at {npy_dir}")
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    # split by file (70/30 default but keep large val pool by design)
    n_train = int(len(files)*0.7)
    train_files = files[:n_train]
    val_files   = files[n_train:]
    print(f"[Info] Files: train={len(train_files)}  val={len(val_files)}")

    # build datasets
    train_ds = ElastoPlastDataset(
        train_files,
        standardize=args.standardize,
        limit=args.limit,
        use_params=args.use_params,
        target_delta=args.target_delta,
        fit_scalers=True,
        x_scaler=None,
        y_scaler=None
    )
    val_ds = ElastoPlastDataset(
        val_files,
        standardize=args.standardize,
        limit=args.limit,
        use_params=args.use_params,
        target_delta=args.target_delta,
        fit_scalers=False,
        x_scaler=train_ds.x_scaler,
        y_scaler=train_ds.y_scaler
    )

    print(f"[Dataset] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    in_dim = train_ds.X.shape[1]
    model = MLP(in_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)

    if args.loss == "huber":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=args.sched_factor, patience=args.sched_patience, min_lr=args.sched_min_lr
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        tl = train_epoch(model, train_loader, opt, loss_fn, device)
        vl, vpred, vtrue = eval_epoch(model, val_loader, loss_fn, device)

        train_hist.append(tl); val_hist.append(vl)
        scheduler.step(vl)

        msg = f"Epoch {epoch}/{args.epochs}  TrainLoss: {tl:.9e}  ValLoss: {vl:.9e}"
        print(msg)

        # Early stopping
        if vl + 1e-12 < best_val:
            best_val = vl
            best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.early_stop and no_improve >= args.early_stop_patience:
                print(f"[EarlyStop] No improve for {no_improve} epochs. Stop at {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # Curves
    plot_loss(train_hist, val_hist, os.path.join(out_dir, "loss_curve.png"))

    # Predictions on one full file (nice curve)
    plot_predictions_single_file(
        model, val_ds, os.path.join(out_dir, "predictions_onefile.png"),
        device, args.target_delta, y_scaler=val_ds.y_scaler, title="Validation (one file)"
    )

    # Validation pool metrics + residuals
    # Use val_ds order-aligned strain proxy for scatter (just take stored raw order)
    # NOTE: loader shuffles disabled -> order preserved
    _, vpred_s, vtrue_s = eval_epoch(model, val_loader, loss_fn, device)
    if val_ds.standardize:
        vpred_s = val_ds.y_scaler.inverse(vpred_s)
        vtrue_s = val_ds.y_scaler.inverse(vtrue_s)

    # Approximate "strain-like" for scatter: reuse the first feature (standardized) to order
    # Better: rebuild from raw to keep it simple: use val_ds.X_raw first column (strain) trimmed to len
    strain_like = val_ds.X_raw[:len(vtrue_s), 0:1]
    plot_residuals(vtrue_s, vpred_s, strain_like, os.path.join(out_dir, "residuals.png"))

    # R2 overall
    r2_all = r2_score(vtrue_s, vpred_s)

    # Elastic vs plastic masks (approx from val_ds.aux, aligned by limit)
    mask_el = val_ds.aux["elastic_mask"][:len(vtrue_s)]
    mask_pl = ~mask_el

    def safe_r2(y, yhat, m):
        if m.sum() < 10: return float("nan")
        return r2_score(y[m], yhat[m])

    r2_el = safe_r2(vtrue_s.reshape(-1), vpred_s.reshape(-1), mask_el)
    r2_pl = safe_r2(vtrue_s.reshape(-1), vpred_s.reshape(-1), mask_pl)

    summary = {
        "in_dim": int(in_dim),
        "use_params": bool(args.use_params),
        "target_delta": bool(args.target_delta),
        "loss": args.loss,
        "train_loss_last": float(train_hist[-1]),
        "val_loss_best": float(best_val),
        "r2_all": float(r2_all),
        "r2_elastic": float(r2_el),
        "r2_plastic": float(r2_pl),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] Saved:")
    print(" ", os.path.join(out_dir, "model.pt"))
    print(" ", os.path.join(out_dir, "loss_curve.png"))
    print(" ", os.path.join(out_dir, "predictions_onefile.png"))
    print(" ", os.path.join(out_dir, "residuals.png"))
    print("Summary:", summary)

if __name__ == "__main__":
    main()
