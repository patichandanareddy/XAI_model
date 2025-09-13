# scripts/run_hyperelastic_v2.py
import os, glob, json, argparse, math
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ----------------------------
# Data loading (robust/flexible)
# ----------------------------
# We accept either:
#  A) many files:  data/hyperelasticity/INPUTS/npy/sample_*.npy  (each a dict)
#  B) single arrays: INPUTS/npy/strain.npy, time.npy (optional), control.npy (optional)
#                    RESULTS/stress.npy
# Keys used when present: 'strain','prev_strain','prev_stress','time','control'
# Optional material parameters (if present): 'mu','k','alpha'
BASE_FEATS = ['strain','prev_strain','prev_stress','time','control']
PARAMS     = ['mu','k','alpha']   # use any that exist

def load_one_npy(path: str) -> Dict:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    elif isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    return d

def load_many_sample_files(root: str, limit: Optional[int]) -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(root,'INPUTS','npy','sample_*.npy')))
    if not files:
        return {}
    Xrows, yrows = [], []
    feat_names_ref: Optional[List[str]] = None
    total = 0
    for f in files:
        d = load_one_npy(f)
        # build features present in this file
        feats = [k for k in BASE_FEATS if k in d]
        # prev_* convenience: if missing create zeros with right shape
        if 'prev_strain' not in d and 'strain' in d:
            ps = np.roll(np.asarray(d['strain']), 1); ps[0] = ps[1]
            d['prev_strain'] = ps
            feats = sorted(set(feats + ['prev_strain']), key=lambda k: BASE_FEATS.index(k))
        if 'prev_stress' not in d and 'stress' in d:
            pt = np.roll(np.asarray(d['stress']), 1); pt[0] = pt[1]
            d['prev_stress'] = pt
            feats = sorted(set(feats + ['prev_stress']), key=lambda k: BASE_FEATS.index(k))
        # optional params if present (constant per sample is fine)
        params_here = [k for k in PARAMS if k in d]
        feats_all = feats + params_here

        Xcols = []
        N = len(d['strain'])
        for k in feats_all:
            v = np.asarray(d[k])
            if np.ndim(v) == 0:
                v = np.full((N,), float(v))
            Xcols.append(v.reshape(-1,1).astype(np.float32))
        X = np.concatenate(Xcols, axis=1) if Xcols else np.zeros((N,0), np.float32)

        y = np.asarray(d['stress'], dtype=np.float32).reshape(-1,1)

        if feat_names_ref is None:
            feat_names_ref = feats_all
        else:
            # align columns to the first sample’s feature order (fill missing with zeros)
            all_feats = feat_names_ref
            X_aligned = []
            for k in all_feats:
                if k in feats_all:
                    idx = feats_all.index(k)
                    X_aligned.append(X[:, idx:idx+1])
                else:
                    X_aligned.append(np.zeros((X.shape[0],1), np.float32))
            X = np.concatenate(X_aligned, axis=1)

        Xrows.append(X); yrows.append(y)
        total += len(X)
        if limit is not None and total >= limit:
            break

    if not Xrows:
        return {}
    X = np.concatenate(Xrows, axis=0)
    y = np.concatenate(yrows, axis=0)
    return dict(X=X, y=y, feature_names=feat_names_ref)

def load_single_arrays(root: str, limit: Optional[int]) -> Dict[str, np.ndarray]:
    inp_dir = os.path.join(root, 'INPUTS', 'npy')
    res_dir = os.path.join(root, 'RESULTS')

    strain_p = os.path.join(inp_dir, 'strain.npy')
    stress_p = os.path.join(res_dir, 'stress.npy')
    if not (os.path.isfile(strain_p) and os.path.isfile(stress_p)):
        return {}

    strain = np.load(strain_p).astype(np.float32).reshape(-1)
    stress = np.load(stress_p).astype(np.float32).reshape(-1)

    feat_names = []
    cols = []

    feat_names.append('strain'); cols.append(strain.reshape(-1,1))

    prev_strain = np.roll(strain, 1); prev_strain[0] = prev_strain[1]
    feat_names.append('prev_strain'); cols.append(prev_strain.reshape(-1,1))

    # prev_stress computed from stress
    prev_stress = np.roll(stress, 1); prev_stress[0] = prev_stress[1]
    feat_names.append('prev_stress'); cols.append(prev_stress.reshape(-1,1))

    # optional extras
    time_p = os.path.join(inp_dir, 'time.npy')
    if os.path.isfile(time_p):
        time = np.load(time_p).astype(np.float32).reshape(-1,1)
        feat_names.append('time'); cols.append(time)

    control_p = os.path.join(inp_dir, 'control.npy')
    if os.path.isfile(control_p):
        control = np.load(control_p).astype(np.float32).reshape(-1,1)
        feat_names.append('control'); cols.append(control)

    X = np.concatenate(cols, axis=1).astype(np.float32)
    y = stress.reshape(-1,1).astype(np.float32)

    if limit is not None:
        X = X[:limit]; y = y[:limit]
    return dict(X=X, y=y, feature_names=feat_names)

def build_dataset(data_dir: str, limit: Optional[int]) -> Dict[str, np.ndarray]:
    d = load_many_sample_files(data_dir, limit)
    if d:
        return d
    d = load_single_arrays(data_dir, limit)
    if d:
        return d
    raise FileNotFoundError(
        f"No data found under {data_dir}. Expected either INPUTS/npy/sample_*.npy "
        f"or INPUTS/npy/strain.npy + RESULTS/stress.npy."
    )

# ----------------------------
# Torch dataset
# ----------------------------
class ArrayDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, standardize: bool, target_delta: bool, feature_names: List[str]):
        self.feature_names = feature_names
        self.standardize = standardize
        self.target_delta = target_delta

        # If target_delta: predict delta_stress := stress - prev_stress (if prev_stress exists)
        if target_delta and 'prev_stress' in feature_names:
            prev_idx = feature_names.index('prev_stress')
            y = y - X[:, prev_idx:prev_idx+1]

        self.X = torch.from_numpy(X.copy())
        self.y = torch.from_numpy(y.reshape(-1,1).copy())

        if standardize:
            self.x_mean = self.X.mean(dim=0, keepdim=True)
            self.x_std  = self.X.std(dim=0, keepdim=True) + 1e-8
            self.X = (self.X - self.x_mean)/self.x_std
        else:
            self.x_mean = None
            self.x_std = None

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128,128,128,128], out_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

# ----------------------------
# Training helpers
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.HuberLoss(delta=1.0)
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device).squeeze(1)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += float(loss) * len(xb)
        n += len(xb)
    return total / max(n,1)

def plot_curve(train_hist: List[float], val_hist: List[float], path: str):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label='train')
    plt.plot(val_hist, label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=140); plt.close()

@torch.no_grad()
def plot_predictions(model: nn.Module, ds: ArrayDS, path_pred: str, path_resid: str, device: torch.device):
    model.eval()
    X = ds.X.to(device); y = ds.y.to(device).squeeze(1)
    pred = model(X)
    # If the model predicted delta, recover stress for plotting when possible
    if ds.target_delta and 'prev_stress' in ds.feature_names:
        prev_idx = ds.feature_names.index('prev_stress')
        prev = ds.X[:, prev_idx].to(device)
        pred_vis = (pred + prev).cpu().numpy()
        y_vis = (y + prev).cpu().numpy()
    else:
        pred_vis = pred.cpu().numpy()
        y_vis = y.cpu().numpy()

    n_plot = min(4000, len(y_vis))
    plt.figure(figsize=(7,4))
    plt.plot(y_vis[:n_plot], label='true', linewidth=1)
    plt.plot(pred_vis[:n_plot], label='pred', linewidth=1)
    plt.legend(); plt.title('Predictions (first chunk)')
    plt.tight_layout(); plt.savefig(path_pred, dpi=140); plt.close()

    # residuals
    resid = (pred_vis.reshape(-1) - y_vis.reshape(-1))
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=60)
    plt.title('Residuals'); plt.tight_layout(); plt.savefig(path_resid, dpi=140); plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='outputs/hyperelastic_v2')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--standardize', action='store_true')
    ap.add_argument('--target_delta', action='store_true', help='Predict Δstress if prev_stress available')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--sched_factor', type=float, default=0.5)
    ap.add_argument('--sched_patience', type=int, default=3)
    ap.add_argument('--sched_min_lr', type=float, default=1e-6)
    ap.add_argument('--early_stop', action='store_true')
    ap.add_argument('--early_stop_patience', type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    data = build_dataset(args.data_dir, args.limit)
    X_np, y_np, feat_names = data['X'], data['y'], data['feature_names']

    # split
    n = len(X_np)
    n_train = int(0.8*n)
    X_train, X_val = X_np[:n_train], X_np[n_train:]
    y_train, y_val = y_np[:n_train], y_np[n_train:]

    train_ds = ArrayDS(X_train, y_train, args.standardize, args.target_delta, feat_names)
    val_ds   = ArrayDS(X_val,   y_val,   args.standardize, args.target_delta, feat_names)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)

    model = MLP(in_dim=train_ds.X.shape[1]).to(device)
    loss_fn = nn.HuberLoss(delta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=args.sched_factor, patience=args.sched_patience, min_lr=args.sched_min_lr
    )

    print(f"[Info] Device: {device.type}, threads: {args.threads}")
    print(f"[Info] Features ({train_ds.X.shape[1]}): {feat_names}")
    print(f"[Dataset] Train={len(train_ds)}  Val={len(val_ds)}  (target_delta={args.target_delta})")

    best_val = math.inf
    best_epoch = -1
    train_hist: List[float] = []
    val_hist: List[float] = []
    patience_used = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0; seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).squeeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss) * len(xb)
            seen += len(xb)
        train_loss = running / max(seen,1)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        train_hist.append(train_loss); val_hist.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            patience_used = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
        else:
            patience_used += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}  TrainLoss: {train_loss:.9e}  ValLoss: {val_loss:.9e}")

        if args.early_stop and patience_used >= args.early_stop_patience:
            print(f"[EarlyStop] no val improvement for {args.early_stop_patience} epochs. Stopping at {epoch}.")
            break

    # load best for reporting/plots
    sd = torch.load(os.path.join(args.out_dir, 'model.pt'), map_location='cpu')
    model.load_state_dict(sd)
    model.to(device).eval()

    # plots
    plot_curve(train_hist, val_hist, os.path.join(args.out_dir, 'loss_curve.png'))
    plot_predictions(model, val_ds, os.path.join(args.out_dir, 'predictions_onefile.png'),
                     os.path.join(args.out_dir, 'residuals.png'), device)

    # summary
    with torch.no_grad():
        y_true = val_ds.y.squeeze(1).numpy()
        y_pred = model(val_ds.X.to(device)).cpu().numpy()
        if val_ds.target_delta and 'prev_stress' in val_ds.feature_names:
            prev = val_ds.X[:, val_ds.feature_names.index('prev_stress')].numpy()
            y_true = y_true + prev; y_pred = y_pred + prev
        r2_all = float(r2_score(y_true, y_pred))
    summary = {
        "in_dim": int(train_ds.X.shape[1]),
        "target_delta": bool(args.target_delta),
        "loss": "huber",
        "train_loss_last": float(train_hist[-1]),
        "val_loss_best": float(best_val),
        "r2_all": r2_all,
        "feature_names": feat_names,
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("[Done] Saved:")
    print(" ", os.path.join(args.out_dir, "model.pt"))
    print(" ", os.path.join(args.out_dir, "loss_curve.png"))
    print(" ", os.path.join(args.out_dir, "predictions_onefile.png"))
    print(" ", os.path.join(args.out_dir, "residuals.png"))
    print("Summary:", summary)

if __name__ == "__main__":
    main()
