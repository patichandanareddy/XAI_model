# scripts/explain_hyper.py
import os, glob, json, argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ----------------------------
# Data discovery (hyperelastic)
# ----------------------------
BASE_FEATS = ['strain','prev_strain','prev_stress','time','control']
PARAM_KEYS = ['mu','k','alpha']  # optional material parameters if present

def load_one_npy(path: str) -> Dict:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    elif isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    return d

def build_xy(d: Dict, use_params: bool, target_delta: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X, y for a single sample dict. Only uses hyperelastic-style fields.
    """
    feats = [k for k in BASE_FEATS if k in d]
    # Create prev_* if missing and we can compute it
    if 'prev_strain' not in d and 'strain' in d:
        ps = np.roll(np.asarray(d['strain']), 1); ps[0] = ps[1]
        d['prev_strain'] = ps
        if 'prev_strain' not in feats:
            feats.append('prev_strain')
    if 'prev_stress' not in d and 'stress' in d:
        pt = np.roll(np.asarray(d['stress']), 1); pt[0] = pt[1]
        d['prev_stress'] = pt
        if 'prev_stress' not in feats:
            feats.append('prev_stress')

    if use_params:
        for k in PARAM_KEYS:
            if k in d:
                feats.append(k)

    # keep a consistent order
    order = [k for k in BASE_FEATS if k in feats] + [k for k in PARAM_KEYS if k in feats]
    feats = order

    N = len(d['strain'])
    X_cols = []
    for k in feats:
        v = np.asarray(d[k])
        if np.ndim(v) == 0:  # scalar → broadcast
            v = np.full((N,), float(v))
        X_cols.append(v.reshape(-1,1).astype(np.float32))
    X = np.concatenate(X_cols, axis=1).astype(np.float32)

    y = np.asarray(d['stress'], dtype=np.float32).reshape(-1,1)
    if target_delta and ('prev_stress' in feats):
        prev = X[:, feats.index('prev_stress'):feats.index('prev_stress')+1]
        y = y - prev

    return X, y, feats

def load_dataset(data_dir: str, max_files: Optional[int], limit: Optional[int], use_params: bool, target_delta: bool):
    """Try multi-sample files first; fall back to single arrays."""
    # A) many sample_*.npy
    files = sorted(glob.glob(os.path.join(data_dir, 'INPUTS', 'npy', 'sample_*.npy')))
    if max_files: files = files[:max_files]
    if files:
        Xs, Ys = [], []
        feat_ref: Optional[List[str]] = None
        total = 0
        for f in tqdm(files, desc='[Load]'):
            d = load_one_npy(f)
            # must have stress + strain for hyperelastic
            if 'stress' not in d or 'strain' not in d:
                continue
            X, y, feats = build_xy(d, use_params, target_delta)
            if feat_ref is None:
                feat_ref = feats
            # align columns to feat_ref
            if feats != feat_ref:
                aligned = []
                for k in feat_ref:
                    if k in feats:
                        aligned.append(X[:, feats.index(k):feats.index(k)+1])
                    else:
                        aligned.append(np.zeros((X.shape[0],1), np.float32))
                X = np.concatenate(aligned, axis=1)
            Xs.append(X); Ys.append(y)
            total += len(X)
            if limit is not None and total >= limit:
                break
        if Xs:
            X = np.concatenate(Xs, axis=0)
            y = np.concatenate(Ys, axis=0)
            if limit is not None and len(X) > limit:
                X = X[:limit]; y = y[:limit]
            return X, y, feat_ref

    # B) single arrays
    inp = os.path.join(data_dir, 'INPUTS', 'npy')
    res = os.path.join(data_dir, 'RESULTS')

    strain_p = os.path.join(inp, 'strain.npy')
    stress_p = os.path.join(res, 'stress.npy')
    if not (os.path.isfile(strain_p) and os.path.isfile(stress_p)):
        raise FileNotFoundError(f"No valid hyperelastic data under {data_dir}")

    strain = np.load(strain_p).astype(np.float32).reshape(-1)
    stress = np.load(stress_p).astype(np.float32).reshape(-1)

    d = {'strain': strain, 'stress': stress}
    # optional
    for k in ['time','control','mu','k','alpha']:
        p = os.path.join(inp, f'{k}.npy')
        if os.path.isfile(p):
            d[k] = np.load(p).astype(np.float32)

    X, y, feats = build_xy(d, use_params, target_delta)
    if limit is not None and len(X) > limit:
        X = X[:limit]; y = y[:limit]
    return X, y, feats

# ----------------------------
# Utilities
# ----------------------------
def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std

def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean)/std

def infer_mlp_from_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    """Rebuild a Sequential MLP from state_dict with keys like net.0.weight / net.2.weight etc."""
    items = [(k, v) for k, v in sd.items() if k.endswith('.weight') and v.ndim == 2]
    if not items:
        raise RuntimeError('No linear weights in state_dict.')

    def layer_idx(name: str) -> int:
        # handle either "net.0.weight" or "0.weight"
        parts = name.split('.')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 9999

    items.sort(key=lambda kv: layer_idx(kv[0]))
    shapes = [t.shape for _, t in items]
    in_dim = shapes[0][1]
    hidden = [s[0] for s in shapes[:-1]]
    out_dim = shapes[-1][0]

    layers: List[nn.Module] = []
    cur = in_dim
    for h in hidden:
        layers += [nn.Linear(cur, h), nn.ReLU()]
        cur = h
    layers += [nn.Linear(cur, out_dim)]
    model = nn.Sequential(*layers)

    # map keys "net.i.weight" -> "i.weight" if needed
    fixed = {}
    for k, v in sd.items():
        if k.startswith('net.'):
            fixed[k[4:]] = v
        else:
            fixed[k] = v
    model.load_state_dict(fixed, strict=True)
    return model

@torch.no_grad()
def perm_importance(model: nn.Module, X: torch.Tensor, y: torch.Tensor, repeats=3, batch=4096):
    model.eval()
    def _r2(xb: torch.Tensor) -> float:
        preds = []
        for i in range(0, len(xb), batch):
            preds.append(model(xb[i:i+batch]))
        p = torch.cat(preds, 0).squeeze(1).cpu().numpy()
        return r2_score(y.cpu().numpy().reshape(-1), p)

    base = _r2(X)
    rng = np.random.default_rng(123)
    imps = []
    X_np = X.cpu().numpy()
    for j in range(X.shape[1]):
        scores = []
        for _ in range(repeats):
            Xp = X_np.copy()
            rng.shuffle(Xp[:, j])
            scores.append(_r2(torch.from_numpy(Xp).to(X.dtype)))
        imps.append(base - float(np.mean(scores)))
    return base, imps

def saliency(model: nn.Module, X: torch.Tensor, batch=2048):
    model.eval()
    n = min(len(X), batch)
    with torch.enable_grad():
        x = X[:n].clone().detach().requires_grad_(True)
        yhat = model(x).sum()
        yhat.backward()
        return x.grad.abs().mean(dim=0).cpu().numpy()

def integrated_gradients(model: nn.Module, X: torch.Tensor, steps=32, batch=2048):
    model.eval()
    n = min(len(X), batch)
    with torch.enable_grad():
        x = X[:n].clone().detach()
        baseline = torch.zeros_like(x)
        total = torch.zeros(x.shape[1], dtype=x.dtype)
        for alpha in torch.linspace(0, 1, steps):
            xi = baseline + alpha*(x - baseline)
            xi.requires_grad_(True)
            yi = model(xi).sum()
            yi.backward()
            total += xi.grad.abs().mean(dim=0).detach()
        return (total / steps).cpu().numpy()

def bar_plot(values: List[float], names: List[str], title: str, path: str):
    order = np.argsort(values)[::-1]
    vals = np.array(values)[order]
    labs = [names[i] for i in order]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labs, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model_path', default='outputs/hyperelastic_v2/model.pt')
    ap.add_argument('--out_dir', default='outputs/hyperelastic_v2')
    ap.add_argument('--standardize', action='store_true')
    ap.add_argument('--max_files', type=int, default=300)
    ap.add_argument('--limit', type=int, default=300000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--permutation_repeats', type=int, default=3)
    ap.add_argument('--use_params', action='store_true')
    ap.add_argument('--target_delta', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_np, y_np, feat_names = load_dataset(
        args.data_dir, args.max_files, args.limit, args.use_params, args.target_delta
    )

    stats = {}
    if args.standardize:
        mean, std = standardize_fit(X_np)
        X_np = standardize_apply(X_np, mean, std)
        stats['x_mean'] = mean.tolist()
        stats['x_std']  = std.tolist()

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np.reshape(-1))

    model = infer_mlp_from_state_dict(torch.load(args.model_path, map_location='cpu')).eval()

    with torch.no_grad():
        preds = []
        for i in range(0, len(X), args.batch):
            preds.append(model(X[i:i+args.batch]))
        yhat = torch.cat(preds, 0).squeeze(1)
    base_r2 = r2_score(y.cpu().numpy(), yhat.cpu().numpy())

    base_r2_perm, imps = perm_importance(model, X, y, repeats=args.permutation_repeats, batch=args.batch)
    sal = saliency(model, X, batch=args.batch)
    ig  = integrated_gradients(model, X, steps=32, batch=args.batch)

    bar_plot(imps, feat_names, 'Permutation Importance', os.path.join(args.out_dir, 'explain_permutation.png'))
    bar_plot(sal.tolist(), feat_names, 'Saliency (|∂y/∂x| avg)', os.path.join(args.out_dir, 'explain_saliency.png'))
    bar_plot(ig.tolist(),  feat_names, 'Integrated Gradients (avg |grad|)', os.path.join(args.out_dir, 'explain_integrated_gradients.png'))

    summary = {
        'feature_names': feat_names,
        'base_r2': float(base_r2),
        'base_r2_perm_call': float(base_r2_perm),
        'permutation_importance': {k: float(v) for k, v in zip(feat_names, imps)},
        'saliency': {k: float(v) for k, v in zip(feat_names, sal.tolist())},
        'integrated_gradients': {k: float(v) for k, v in zip(feat_names, ig.tolist())},
        'standardized': bool(args.standardize),
        'use_params': bool(args.use_params),
        'target_delta': bool(args.target_delta),
        'n_samples': int(len(X_np))
    }
    with open(os.path.join(args.out_dir, 'explain_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("[Explain] Saved:")
    print(" ", os.path.join(args.out_dir, 'explain_permutation.png'))
    print(" ", os.path.join(args.out_dir, 'explain_saliency.png'))
    print(" ", os.path.join(args.out_dir, 'explain_integrated_gradients.png'))
    print(" ", os.path.join(args.out_dir, 'explain_summary.json'))

if __name__ == '__main__':
    main()
