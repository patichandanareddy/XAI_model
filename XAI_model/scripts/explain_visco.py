# scripts/explain_visco.py
import argparse, os, json, glob
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Preferred feature keys for typical viscoelastic data.
# We will only use keys that actually exist in each npy dict.
PREFERRED_FEATURES = [
    "strain", "prev_strain", "prev_stress",
    "e_visco", "prev_e_visco",
    "control", "time_step", "rate"
]

# Common parameter keys (if present); we’ll include them with --use_params
PREFERRED_PARAMS = ["E", "eta", "H_kin", "sig0"]

def load_one_npy(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    elif isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    return d

def pick_feature_keys(d: dict, use_params: bool) -> List[str]:
    keys = [k for k in PREFERRED_FEATURES if k in d]
    if use_params:
        keys += [k for k in PREFERRED_PARAMS if k in d]
    # If nothing matched, fallback to all 1D numeric arrays except stress-like targets
    if not keys:
        for k, v in d.items():
            if k.lower() in ("stress", "delta_stress", "target"):
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                v = np.asarray(v)
                if v.ndim == 1 and np.issubdtype(v.dtype, np.number):
                    keys.append(k)
    return keys

def build_xy(d: dict, feat_keys: List[str], target_delta: bool):
    N = len(d[feat_keys[0]]) if feat_keys else len(d["strain"])
    X_cols = []
    for k in feat_keys:
        v = d[k]
        v = np.asarray(v)
        if v.ndim == 0:
            v = np.full((N,), float(v))
        X_cols.append(v.reshape(-1, 1).astype(np.float32))
    X = np.concatenate(X_cols, axis=1).astype(np.float32)

    # y target
    if "stress" in d:
        y = np.asarray(d["stress"], dtype=np.float32).reshape(-1, 1)
    elif "target" in d:
        y = np.asarray(d["target"], dtype=np.float32).reshape(-1, 1)
    else:
        raise KeyError("Could not find 'stress' or 'target' in npy file.")

    if target_delta:
        # subtract previous stress if available
        if "prev_stress" in d:
            y = y - np.asarray(d["prev_stress"], dtype=np.float32).reshape(-1, 1)
        elif "delta_stress" in d:
            y = np.asarray(d["delta_stress"], dtype=np.float32).reshape(-1, 1)
        # else we’ll just keep y as-is

    return X, y

def load_dataset(data_dir: str, max_files: int, limit: int, use_params: bool, target_delta: bool):
    files = sorted(glob.glob(os.path.join(data_dir, "INPUTS", "npy", "sample_*.npy")))
    if not files:
        raise FileNotFoundError(f"No files at {data_dir}/INPUTS/npy/sample_*.npy")
    if max_files is not None:
        files = files[:max_files]

    Xs, Ys = [], []
    feat_names = None
    total = 0
    for f in tqdm(files, desc="[Load]"):
        d = load_one_npy(f)
        feats = pick_feature_keys(d, use_params)
        if not feats:
            # skip if truly nothing to use
            continue
        X, y = build_xy(d, feats, target_delta)
        if feat_names is None:
            feat_names = feats
        Xs.append(X); Ys.append(y)
        total += len(X)
        if limit is not None and total >= limit:
            break

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    if limit is not None and len(X) > limit:
        X = X[:limit]; y = y[:limit]
    return X, y, feat_names

def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std

def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std

def strip_prefix_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handle both "net.X.weight" and plain "X.weight"
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            new_sd[k[4:]] = v
        else:
            new_sd[k] = v
    return new_sd

def infer_mlp_from_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    sd = strip_prefix_state_dict(sd)
    items = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    if not items:
        raise RuntimeError("No linear weights in state_dict.")
    # sort by integer index in keys "0.weight", "2.weight"...
    def idx_of(name):
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 9999
    items.sort(key=lambda kv: idx_of(kv[0]))
    shapes = [t.shape for _, t in items]
    in_dim = shapes[0][1]
    hidden = [s[0] for s in shapes[:-1]]
    out_dim = shapes[-1][0]
    layers, cur_in = [], in_dim
    for h in hidden:
        layers += [nn.Linear(cur_in, h), nn.ReLU()]
        cur_in = h
    layers += [nn.Linear(cur_in, out_dim)]
    model = nn.Sequential(*layers)
    model.load_state_dict(sd, strict=False)
    return model

def try_load_model(path: str) -> nn.Module:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError("Unsupported checkpoint format.")
    return infer_mlp_from_state_dict(sd)

@torch.no_grad()
def perm_importance(model: nn.Module, X: torch.Tensor, y: torch.Tensor, repeats=3, batch=4096):
    model.eval()
    def _pred(Xt):
        preds = []
        for i in range(0, len(Xt), batch):
            preds.append(model(Xt[i:i+batch]))
        return torch.cat(preds, 0).squeeze(1)
    base = r2_score(y.cpu().numpy().reshape(-1), _pred(X).cpu().numpy().reshape(-1))
    rng = np.random.default_rng(123)
    imps = []
    X_np = X.cpu().numpy()
    for j in range(X.shape[1]):
        scores = []
        for _ in range(repeats):
            Xp = X_np.copy()
            rng.shuffle(Xp[:, j])
            pr = _pred(torch.from_numpy(Xp).to(X.dtype))
            scores.append(r2_score(y.cpu().numpy().reshape(-1), pr.cpu().numpy().reshape(-1)))
        imps.append(base - float(np.mean(scores)))
    return base, imps

def saliency(model: nn.Module, X: torch.Tensor, batch=2048):
    model.eval()
    n = min(len(X), batch)
    x = X[:n].clone().detach().requires_grad_(True)
    yhat = model(x).sum()
    yhat.backward()
    return x.grad.abs().mean(dim=0).cpu().numpy()

def integrated_gradients(model: nn.Module, X: torch.Tensor, steps=32, batch=2048):
    model.eval()
    n = min(len(X), batch)
    x = X[:n].clone().detach()
    baseline = torch.zeros_like(x)
    total = torch.zeros(x.shape[1], dtype=x.dtype)
    for alpha in torch.linspace(0, 1, steps):
        xi = (baseline + alpha * (x - baseline)).requires_grad_(True)
        yi = model(xi).sum()
        yi.backward()
        total += xi.grad.abs().mean(dim=0).detach()
    return (total / steps).cpu().numpy()

def bar_plot(values: List[float], names: List[str], title: str, path: str):
    order = np.argsort(values)[::-1]
    vals = np.array(values)[order]
    labs = [names[i] for i in order]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labs, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

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
    ap.add_argument("--use_params", action="store_true")
    ap.add_argument("--target_delta", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_np, y_np, feat_names = load_dataset(
        args.data_dir, args.max_files, args.limit, args.use_params, args.target_delta
    )
    if feat_names is None:
        raise RuntimeError("Could not infer feature names from data.")

    stats = {}
    if args.standardize:
        mean, std = standardize_fit(X_np)
        X_np = standardize_apply(X_np, mean, std)
        stats["x_mean"] = mean.tolist()
        stats["x_std"] = std.tolist()

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np.reshape(-1))

    model = try_load_model(args.model_path).eval()
    _ = model(X[:2])  # warm up

    with torch.no_grad():
        preds = []
        for i in range(0, len(X), args.batch):
            preds.append(model(X[i:i+args.batch]))
        yhat = torch.cat(preds, 0).squeeze(1)
    base_r2 = r2_score(y.cpu().numpy(), yhat.cpu().numpy())

    base_r2_perm, imps = perm_importance(model, X, y, repeats=args.permutation_repeats, batch=args.batch)
    sal = saliency(model, X, batch=args.batch)
    ig  = integrated_gradients(model, X, steps=32, batch=args.batch)

    bar_plot(imps, feat_names, "Permutation Importance", os.path.join(args.out_dir, "explain_permutation.png"))
    bar_plot(sal.tolist(), feat_names, "Saliency (|∂y/∂x| avg)", os.path.join(args.out_dir, "explain_saliency.png"))
    bar_plot(ig.tolist(),  feat_names, "Integrated Gradients (avg |grad|)", os.path.join(args.out_dir, "explain_integrated_gradients.png"))

    summary = {
        "feature_names": feat_names,
        "base_r2": float(base_r2),
        "permutation_importance": {k: float(v) for k, v in zip(feat_names, imps)},
        "saliency": {k: float(v) for k, v in zip(feat_names, sal.tolist())},
        "integrated_gradients": {k: float(v) for k, v in zip(feat_names, ig.tolist())},
        "standardized": bool(args.standardize),
        "use_params": bool(args.use_params),
        "target_delta": bool(args.target_delta),
        "n_samples": int(len(X_np))
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
