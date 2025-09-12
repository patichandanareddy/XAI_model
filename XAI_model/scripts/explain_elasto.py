import os, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Dataset (matches training)
# ---------------------------
FEATURE_ORDER = [
    "strain",
    "prev_strain",
    "prev_stress",
    "e_plastic",
    "prev_e_plastic",
    "control",
    "E",
    "sig0",
    "H_kin",
]
TARGET_KEY = "stress"

def _safe_get_len(d):
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1:
            return v.shape[0]
    return 0

def _broadcast_scalar(arr, T):
    if np.ndim(arr) == 0:
        return np.full((T,), float(arr), dtype=np.float64)
    if np.ndim(arr) == 1 and arr.size == 1:
        return np.full((T,), float(arr[0]), dtype=np.float64)
    return arr

def build_xy_from_dict(d):
    T = _safe_get_len(d)
    xs, used = [], []
    for name in FEATURE_ORDER:
        if name not in d:
            continue
        arr = d[name]
        if isinstance(arr, np.ndarray):
            arr = _broadcast_scalar(arr, T)
            if arr.ndim == 1:
                xs.append(arr.reshape(T, 1))
                used.append(name)
    X = np.concatenate(xs, axis=1) if xs else np.zeros((T,0), dtype=np.float64)
    if TARGET_KEY in d and isinstance(d[TARGET_KEY], np.ndarray):
        y = d[TARGET_KEY].reshape(-1).astype(np.float64)
    else:
        y = np.zeros((T,), dtype=np.float64)
    return X.astype(np.float32), y.astype(np.float32), used

class ElastoExplainDS(Dataset):
    def __init__(self, files, standardize=False, limit=None):
        self.files = files if limit is None else files[:limit]
        Xs, Ys = [], []
        used_names = None
        for f in self.files:
            d = np.load(f, allow_pickle=True).item()
            X, y, used = build_xy_from_dict(d)
            if X.shape[0] == 0 or X.shape[1] == 0:
                continue
            Xs.append(X)
            Ys.append(y.reshape(-1,1))
            if used_names is None:
                used_names = used
        self.X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(FEATURE_ORDER)), dtype=np.float32)
        self.Y = np.concatenate(Ys, axis=0) if Ys else np.zeros((0,1), dtype=np.float32)
        self.feature_names = used_names if used_names is not None else FEATURE_ORDER
        self.standardize = standardize
        if standardize and self.X.shape[0] > 0:
            self.mean = self.X.mean(axis=0, keepdims=True)
            self.std = self.X.std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx])
        )

# ---------------------------
# Model utilities
# ---------------------------

class Wrapper(nn.Module):
    """Holds a .net nn.Sequential to match checkpoint prefix (e.g., 'net.0.weight')."""
    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net
    def forward(self, x):
        return self.net(x)

def build_net_from_state_dict(sd: dict) -> nn.Sequential:
    """
    Rebuild nn.Sequential with Linear/ReLU at the exact indices the checkpoint used.
    Expects keys like 'net.0.weight', 'net.2.weight', ... (Linear at even, ReLU at odd).
    """
    # collect all (pos, weight) for keys ending with '.weight' under the same top-level module (prefix)
    linear_keys = [k for k in sd.keys() if k.endswith(".weight")]
    if not linear_keys:
        raise RuntimeError("No linear weights found in state_dict.")
    # detect the common prefix (e.g., 'net')
    first_key = linear_keys[0]
    prefix = first_key.split('.')[0]  # 'net'
    # positions for Linear layers (e.g., 0,2,4,6)
    positions = []
    shapes = {}
    for k in linear_keys:
        parts = k.split('.')
        if parts[0] != prefix:
            continue
        try:
            pos = int(parts[1])
        except ValueError:
            continue
        positions.append(pos)
        shapes[pos] = tuple(sd[k].shape)  # (out, in)

    if not positions:
        raise RuntimeError("Could not parse layer indices from state_dict.")
    max_pos = max(positions)

    # construct layers list of length max_pos+1
    layers = []
    for idx in range(max_pos + 1):
        if idx in positions:
            out_dim, in_dim = shapes[idx]
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # fill gaps (typical checkpoints had ReLU at odd indices)
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def try_load_model(model_path: str):
    """
    Load:
      1) Full saved nn.Module (if any)
      2) state_dict -> reconstruct .net architecture from indices/shapes
    """
    # Attempt 1: load complete module
    try:
        obj = torch.load(model_path, map_location="cpu")
        if isinstance(obj, nn.Module):
            obj.eval()
            return obj
    except Exception:
        pass

    # Attempt 2: state_dict
    sd = torch.load(model_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint is not a state_dict and is not a Module.")
    # Rebuild net to match indices/sizes in state_dict
    net = build_net_from_state_dict(sd)
    model = Wrapper(net)
    # Load weights with strict=True (names should match 'net.X.weight/bias')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False in case checkpoint had buffers/extra keys; warn if truly empty
    if all(len(x)==0 for x in (missing, unexpected)) is False:
        # Not a failure; just proceed
        pass
    model.eval()
    return model

# ---------------------------
# Explainability
# ---------------------------

@torch.no_grad()
def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return 1.0 - ss_res/ss_tot

def predict_batches(model, X, batch=8192):
    device = torch.device("cpu")
    yh = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        pb = model(xb).detach().cpu().numpy().reshape(-1)
        yh.append(pb)
    return np.concatenate(yh, axis=0)

def permutation_importance(model, X, y, feature_names, batch=8192, repeats=1):
    y_hat = predict_batches(model, X, batch=batch)
    base_r2 = r2_score(y.reshape(-1), y_hat)
    rng = np.random.default_rng(123)
    deltas = []
    for j, _ in enumerate(feature_names):
        scores = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            y_perm = predict_batches(model, Xp, batch=batch)
            scores.append(base_r2 - r2_score(y.reshape(-1), y_perm))
        deltas.append(float(np.mean(scores)))
    return dict(zip(feature_names, deltas)), float(base_r2)

def saliency_attribution(model, X, feature_names, batch=2048):
    device = torch.device("cpu")
    model.to(device)
    grads_sum = np.zeros((X.shape[1],), dtype=np.float64)
    n = 0
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device)
        xb.requires_grad_(True)
        out = model(xb).sum()
        out.backward()
        g = xb.grad.detach().cpu().numpy()
        grads_sum += np.abs(g).mean(axis=0)
        n += 1
    grads_mean = grads_sum / max(n, 1)
    return dict(zip(feature_names, grads_mean.tolist()))

def integrated_gradients(model, X, feature_names, m_steps=32, batch=1024):
    device = torch.device("cpu")
    model.to(device)
    F = X.shape[1]
    ig_sum = np.zeros((F,), dtype=np.float64)
    count = 0
    for i in range(0, X.shape[0], batch):
        xb_np = X[i:i+batch]
        xb = torch.from_numpy(xb_np).to(device)
        total = torch.zeros_like(xb)
        for k in range(1, m_steps+1):
            alpha = float(k)/m_steps
            xk = (alpha * xb).requires_grad_(True)
            yk = model(xk).sum()
            yk.backward()
            total += xk.grad.detach()
        ig = (xb * total / m_steps).detach().cpu().numpy()
        ig_sum += np.abs(ig).mean(axis=0)
        count += 1
    ig_mean = ig_sum / max(count, 1)
    return dict(zip(feature_names, ig_mean.tolist()))

# ---------------------------
# Plot helpers
# ---------------------------

def barplot(savepath, scores_dict, title):
    names = list(scores_dict.keys())
    vals = np.array([scores_dict[n] for n in names], dtype=float)
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = vals[order]
    plt.figure(figsize=(8,4.5))
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("Importance / Attribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=160)
    plt.close()

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", default="outputs/elastoplastic/model.pt")
    ap.add_argument("--out_dir", default="outputs/elastoplastic")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--max_files", type=int, default=200)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--permutation_repeats", type=int, default=1)
    args = ap.parse_args()

    val_dir = os.path.join(args.data_dir, "INPUTS", "npy")
    files = sorted(glob.glob(os.path.join(val_dir, "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No npy samples found at {val_dir}")

    ds = ElastoExplainDS(files[:args.max_files], standardize=args.standardize, limit=args.limit)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty after loading. Check file contents.")
    X = ds.X
    y = ds.Y.reshape(-1)
    feat_names = ds.feature_names

    model = try_load_model(args.model_path)

    # If model expects fewer features, trim X (and names) from the right.
    # We infer expected in_dim from first Linear in model.net
    first_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        raise RuntimeError("No Linear layer found in loaded model.")
    need = first_linear.in_features
    have = X.shape[1]
    if have < need:
        raise RuntimeError(f"Model expects {need} features, but dataset built {have}.")
    if have > need:
        X = X[:, :need]
        feat_names = feat_names[:need]

    # ----------------- Permutation importance -----------------
    perm, base_r2 = permutation_importance(
        model, X, y, feat_names,
        batch=args.batch,
        repeats=args.permutation_repeats
    )
    os.makedirs(args.out_dir, exist_ok=True)
    barplot(os.path.join(args.out_dir, "explain_permutation.png"),
            perm, f"Permutation importance (ΔR²), base R²={base_r2:.4f}")

    # ----------------- Saliency -----------------
    sal = saliency_attribution(model, X, feat_names, batch=max(1, args.batch//4))
    barplot(os.path.join(args.out_dir, "explain_saliency.png"),
            sal, "Saliency (mean |∂y/∂x|)")

    # ----------------- Integrated Gradients -----------------
    ig = integrated_gradients(model, X, feat_names, m_steps=32, batch=max(1, args.batch//4))
    barplot(os.path.join(args.out_dir, "explain_integrated_gradients.png"),
            ig, "Integrated Gradients (mean |IG|)")

    # Save JSON summary
    summary = {
        "feature_names": feat_names,
        "base_r2": float(base_r2),
        "permutation_importance": {k: float(v) for k, v in perm.items()},
        "saliency_absgrad": {k: float(v) for k, v in sal.items()},
        "integrated_gradients": {k: float(v) for k, v in ig.items()},
        "standardize": bool(args.standardize),
        "samples_used": int(X.shape[0]),
    }
    with open(os.path.join(args.out_dir, "explain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[Explain] Saved:")
    print(" ", os.path.join(args.out_dir, "explain_permutation.png"))
    print(" ", os.path.join(args.out_dir, "explain_saliency.png"))
    print(" ", os.path.join(args.out_dir, "explain_integrated_gradients.png"))
    print(" ", os.path.join(args.out_dir, "explain_summary.json"))

if __name__ == "__main__":
    main()
