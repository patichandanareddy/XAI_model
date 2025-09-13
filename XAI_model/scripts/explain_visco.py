import os, json, glob, argparse
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------------------------
# Data + features
# ---------------------------
def load_one_npy(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    elif isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    return d

def build_xy(d: dict, feature_names: List[str], target_delta: bool) -> Tuple[np.ndarray, np.ndarray]:
    if "strain" not in d:
        raise KeyError(f"'strain' not in sample keys: {list(d.keys())}")
    N = len(d["strain"])
    cols = []
    for k in feature_names:
        if k not in d:
            # If a feature is missing for a particular file, fill with zeros of correct length
            v = np.zeros((N,), dtype=np.float32)
        else:
            v = d[k]
            if np.ndim(v) == 0:
                v = np.full((N,), float(v))
        cols.append(np.asarray(v, dtype=np.float32).reshape(-1,1))
    X = np.concatenate(cols, axis=1).astype(np.float32)

    y = np.asarray(d["stress"], dtype=np.float32).reshape(-1,1)
    if target_delta and "prev_stress" in d:
        y = y - np.asarray(d["prev_stress"], dtype=np.float32).reshape(-1,1)
    return X, y

def load_dataset(data_dir: str, feature_names: List[str], max_files: int=None, limit: int=None,
                 target_delta: bool=True, standardize: bool=False):
    files = sorted(glob.glob(os.path.join(data_dir, "INPUTS", "npy", "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No files under {data_dir}/INPUTS/npy/sample_*.npy")
    if max_files:
        files = files[:max_files]

    Xs, Ys = [], []
    total = 0
    for f in tqdm(files, desc="[Load]"):
        d = load_one_npy(f)
        X, y = build_xy(d, feature_names, target_delta=target_delta)
        Xs.append(X); Ys.append(y)
        total += len(X)
        if limit is not None and total >= limit:
            break

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)

    if limit is not None and len(X) > limit:
        X = X[:limit]; y = y[:limit]

    stats = {}
    if standardize:
        mean = X.mean(axis=0, keepdims=True)
        std  = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean)/std
        stats["x_mean"] = mean.tolist()
        stats["x_std"]  = std.tolist()

    return torch.from_numpy(X), torch.from_numpy(y.reshape(-1)), stats

# ---------------------------
# Model loading
# ---------------------------
def infer_mlp_from_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    # accept both "net.*" and plain indices ("0.weight")
    key_map = {}
    for k in list(sd.keys()):
        if k.startswith("net."):
            key_map[k.replace("net.", "")] = sd.pop(k)
        else:
            key_map[k] = sd.pop(k)
    sd = key_map  # now keys like "0.weight", "2.bias", etc.

    items = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    if not items:
        raise RuntimeError("No linear weights in state_dict.")

    def idx_of(name):
        # "0.weight" -> 0 , "2.weight" -> 2, etc.
        try:
            return int(name.split(".")[0])
        except:
            return 9999

    items.sort(key=lambda kv: idx_of(kv[0]))
    shapes = [t.shape for _, t in items]
    in_dim = shapes[0][1]
    hidden = [s[0] for s in shapes[:-1]]
    out_dim = shapes[-1][0]

    layers = []
    cur = in_dim
    for h in hidden:
        layers += [nn.Linear(cur, h), nn.ReLU()]
        cur = h
    layers += [nn.Linear(cur, out_dim)]
    model = nn.Sequential(*layers)

    # rebuild a clean state_dict with new names "0.weight", etc.
    clean = {}
    for k, v in items:
        base = k.split(".")[0]
        clean[f"{base}.weight"] = v
        bkey = f"{base}.bias"
        if bkey in sd:
            clean[bkey] = sd[bkey]
    # also include the final output layer bias if present
    last_idx = max([idx_of(k) for k in clean.keys()])
    out_b = f"{last_idx}.bias"
    if out_b in sd:
        clean[out_b] = sd[out_b]

    model.load_state_dict(clean, strict=False)
    return model

def try_load_model(path: str) -> nn.Module:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError("Unsupported checkpoint format")
    return infer_mlp_from_state_dict(sd).eval()

# ---------------------------
# XAI
# ---------------------------
@torch.no_grad()
def perm_importance(model: nn.Module, X: torch.Tensor, y: torch.Tensor, repeats=3, batch=4096):
    model.eval()
    def _preds(xb):
        out = []
        for i in range(0, len(xb), batch):
            out.append(model(xb[i:i+batch]))
        return torch.cat(out, 0)

    base = r2_score(y.cpu().numpy(), _preds(X).cpu().numpy().reshape(-1))
    rng = np.random.default_rng(123)
    X_np = X.cpu().numpy()
    imps = []
    for j in range(X.shape[1]):
        scores = []
        for _ in range(repeats):
            Xp = X_np.copy()
            rng.shuffle(Xp[:, j])
            yp = _preds(torch.from_numpy(Xp).to(X.dtype)).cpu().numpy().reshape(-1)
            scores.append(r2_score(y.cpu().numpy(), yp))
        imps.append(base - float(np.mean(scores)))
    return base, imps

def saliency(model: nn.Module, X: torch.Tensor, batch=2048):
    model.eval()
    n = min(len(X), batch)
    x = X[:n].clone().detach()
    x.requires_grad_(True)
    yhat = model(x).sum()
    yhat.backward()
    g = x.grad.abs().mean(dim=0).detach().cpu().numpy()
    return g

def integrated_gradients(model: nn.Module, X: torch.Tensor, steps=32, batch=2048):
    model.eval()
    n = min(len(X), batch)
    x = X[:n].clone().detach()
    baseline = torch.zeros_like(x)
    total = torch.zeros(x.shape[1], dtype=x.dtype)
    for alpha in torch.linspace(0, 1, steps):
        xi = baseline + alpha*(x - baseline)
        xi.requires_grad_(True)
        yi = model(xi).sum()
        yi.backward()
        total += xi.grad.abs().mean(dim=0).detach()
    return (total/steps).cpu().numpy()

def bar_plot(values: List[float], names: List[str], title: str, path: str):
    order = np.argsort(values)[::-1]
    vals = np.array(values)[order]
    labs = [names[i] for i in order]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labs, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_path", default="outputs/viscoelastic_v2/model.pt")
    ap.add_argument("--out_dir", default="outputs/viscoelastic_v2")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--max_files", type=int, default=300)
    ap.add_argument("--limit", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--permutation_repeats", type=int, default=3)
    ap.add_argument("--features_json", default="outputs/viscoelastic_v2/used_features.json",
                    help="Path to feature list saved by training (used_features.json).")
    ap.add_argument("--target_delta", action="store_true", help="Use Δstress target (match training).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the exact features used by training
    if not os.path.exists(args.features_json):
        raise FileNotFoundError(f"Missing {args.features_json}. Train first to create it.")
    feat_names = json.load(open(args.features_json))
    if not isinstance(feat_names, list) or not feat_names:
        raise RuntimeError("Invalid feature list in used_features.json")

    # Data
    X, y, stats = load_dataset(
        args.data_dir, feat_names,
        max_files=args.max_files, limit=args.limit,
        target_delta=args.target_delta, standardize=args.standardize
    )

    # Model
    model = try_load_model(args.model_path).eval()
    # quick forward to ensure shape compatibility
    with torch.no_grad():
        _ = model(X[:2])

    # Metrics + XAI
    with torch.no_grad():
        yhat = []
        for i in range(0, len(X), args.batch):
            yhat.append(model(X[i:i+args.batch]))
        yhat = torch.cat(yhat, 0).squeeze(1)
    base_r2 = r2_score(y.cpu().numpy(), yhat.cpu().numpy())

    base_r2_perm, imps = perm_importance(model, X, y, repeats=args.permutation_repeats, batch=args.batch)
    sal = saliency(model, X, batch=args.batch)
    ig  = integrated_gradients(model, X, steps=32, batch=args.batch)

    # Plots
    bar_plot(imps, feat_names, "Permutation Importance", os.path.join(args.out_dir, "explain_permutation.png"))
    bar_plot(sal.tolist(), feat_names, "Saliency (|∂y/∂x| avg)", os.path.join(args.out_dir, "explain_saliency.png"))
    bar_plot(ig.tolist(), feat_names, "Integrated Gradients (avg |grad|)", os.path.join(args.out_dir, "explain_integrated_gradients.png"))

    # Summary
    summary = {
        "feature_names": feat_names,
        "base_r2": float(base_r2),
        "base_r2_perm_call": float(base_r2_perm),
        "permutation_importance": {k: float(v) for k, v in zip(feat_names, imps)},
        "saliency": {k: float(v) for k, v in zip(feat_names, sal.tolist())},
        "integrated_gradients": {k: float(v) for k, v in zip(feat_names, ig.tolist())},
        "standardized": bool(args.standardize),
        "target_delta": bool(args.target_delta),
        "n_samples": int(len(X)),
        "x_stats": stats,
    }
    with open(os.path.join(args.out_dir, "explain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[Explain] Saved:")
    for k in ["explain_permutation.png","explain_saliency.png","explain_integrated_gradients.png","explain_summary.json"]:
        print(" ", os.path.join(args.out_dir, k))

if __name__ == "__main__":
    main()
