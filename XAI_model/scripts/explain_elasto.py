#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# ----------------------------
# Feature configuration
# ----------------------------
FEATURES6 = ['strain', 'prev_strain', 'prev_stress',
             'e_plastic', 'prev_e_plastic', 'control']
PARAMS3   = ['E', 'sig0', 'H_kin']  # optional material params


# ----------------------------
# Data loading helpers
# ----------------------------
def load_one_npy(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    # Support .npy dict saved as 0-d array, or .npz files
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    elif isinstance(d, np.lib.npyio.NpzFile):
        d = dict(d.items())
    return d


def build_xy(d: dict, use_params: bool, target_delta: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats = FEATURES6.copy()
    if use_params:
        feats += PARAMS3

    X_cols = []
    N = len(d['strain'])
    for k in feats:
        v = d[k]
        # allow scalars for E/sig0/H_kin
        if np.ndim(v) == 0:
            v = np.full((N,), float(v), dtype=np.float32)
        X_cols.append(np.asarray(v, dtype=np.float32).reshape(-1, 1))

    X = np.concatenate(X_cols, axis=1).astype(np.float32)

    y = np.asarray(d['stress'], dtype=np.float32).reshape(-1, 1)
    if target_delta:
        prev = np.asarray(d['prev_stress'], dtype=np.float32).reshape(-1, 1)
        y = y - prev

    return X, y, feats


def load_dataset(
    data_dir: str,
    max_files: int,
    limit: int,
    use_params: bool,
    target_delta: bool
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    files = sorted(glob.glob(os.path.join(data_dir, 'INPUTS', 'npy', 'sample_*.npy')))
    if not files:
        raise FileNotFoundError(f'No files under {data_dir}/INPUTS/npy/sample_*.npy')
    if max_files is not None:
        files = files[:max_files]

    Xs, Ys = [], []
    feats_ref: List[str] = None
    total = 0

    for f in tqdm(files, desc='[Load]'):
        d = load_one_npy(f)
        X, y, feats = build_xy(d, use_params, target_delta)
        if feats_ref is None:
            feats_ref = feats
        Xs.append(X)
        Ys.append(y)
        total += len(X)
        if limit is not None and total >= limit:
            break

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    if limit is not None and len(X) > limit:
        X = X[:limit]
        y = y[:limit]

    return X, y, feats_ref


# ----------------------------
# Standardization helpers
# ----------------------------
def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std

def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


# ----------------------------
# Model loading (shape-/key-agnostic)
# ----------------------------
def _normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrappers like 'module.' and 'net.' from state_dict keys."""
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("net."):
            k2 = k2[len("net."):]
        new_sd[k2] = v
    return new_sd


def infer_mlp_from_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Infer a plain MLP (Sequential of Linear/ReLU.../Linear) from linear weight shapes
    and load the (normalized) state_dict into it.
    """
    sd = _normalize_state_dict_keys(sd)

    # Collect linear layers in order (keys like "0.weight", "2.weight", ...)
    items = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    if not items:
        raise RuntimeError("No linear layer weights found in the state_dict.")

    def _layer_index(name: str) -> int:
        # Extract the first numeric component from names like "0.weight", "2.bias"
        for part in name.split("."):
            if part.isdigit():
                return int(part)
        return 10_000

    items.sort(key=lambda kv: _layer_index(kv[0]))
    shapes = [w.shape for _, w in items]
    in_dim = shapes[0][1]
    hidden = [s[0] for s in shapes[:-1]]
    out_dim = shapes[-1][0]

    layers: List[nn.Module] = []
    cur_in = in_dim
    for h in hidden:
        layers += [nn.Linear(cur_in, h), nn.ReLU()]
        cur_in = h
    layers += [nn.Linear(cur_in, out_dim)]

    model = nn.Sequential(*layers)

    # make sure keys are normalized to match the Sequential indices
    sd = _normalize_state_dict_keys(sd)
    model.load_state_dict(sd, strict=True)
    return model


def try_load_model(path: str) -> nn.Module:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError("Unsupported checkpoint format.")
    sd = _normalize_state_dict_keys(sd)
    return infer_mlp_from_state_dict(sd)


# ----------------------------
# Explainability methods
# ----------------------------
@torch.no_grad()
def perm_importance(model: nn.Module, X: torch.Tensor, y: torch.Tensor, repeats=3, batch=4096):
    model.eval()

    def _predict_batches(Xt: torch.Tensor) -> np.ndarray:
        preds = []
        for i in range(0, len(Xt), batch):
            preds.append(model(Xt[i:i+batch]))
        return torch.cat(preds, 0).squeeze(1).cpu().numpy()

    # baseline R^2
    base_preds = _predict_batches(X)
    base = r2_score(y.cpu().numpy().reshape(-1), base_preds)

    rng = np.random.default_rng(123)
    imps = []
    X_np = X.cpu().numpy()
    for j in range(X.shape[1]):
        scores = []
        for _ in range(repeats):
            Xp = X_np.copy()
            rng.shuffle(Xp[:, j])
            p = _predict_batches(torch.from_numpy(Xp).to(X.dtype))
            scores.append(r2_score(y.cpu().numpy().reshape(-1), p))
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
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)
        yi = model(xi).sum()
        yi.backward()
        total += xi.grad.abs().mean(dim=0).detach()
    return (total / steps).cpu().numpy()


# ----------------------------
# Plotting
# ----------------------------
def bar_plot(values: List[float], names: List[str], title: str, path: str):
    order = np.argsort(values)[::-1]
    vals = np.array(values)[order]
    labs = [names[i] for i in order]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labs, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model_path', default='outputs/elastoplastic/model.pt')
    ap.add_argument('--out_dir', default='outputs/elastoplastic')
    ap.add_argument('--standardize', action='store_true')
    ap.add_argument('--max_files', type=int, default=200)
    ap.add_argument('--limit', type=int, default=200000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--permutation_repeats', type=int, default=3)
    # v2-compatible flags
    ap.add_argument('--use_params', action='store_true')
    ap.add_argument('--target_delta', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load & (optionally) standardize inputs
    X_np, y_np, feat_names = load_dataset(
        args.data_dir, args.max_files, args.limit,
        args.use_params, args.target_delta
    )

    stats = {}
    if args.standardize:
        mean, std = standardize_fit(X_np)
        X_np = standardize_apply(X_np, mean, std)
        stats['x_mean'] = mean.tolist()
        stats['x_std']  = std.tolist()

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np.reshape(-1))

    # Load model (agnostic to 'net.' / 'module.' prefixes)
    model = try_load_model(args.model_path).eval()

    # Warm-up call to ensure shapes are fine
    with torch.no_grad():
        _ = model(X[:2])

    # Base R^2
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), args.batch):
            preds.append(model(X[i:i+args.batch]))
        yhat = torch.cat(preds, 0).squeeze(1)
    base_r2 = r2_score(y.cpu().numpy(), yhat.cpu().numpy())

    # Permutation, saliency, IG
    base_r2_perm, imps = perm_importance(model, X, y, repeats=args.permutation_repeats, batch=args.batch)
    sal = saliency(model, X, batch=args.batch)
    ig  = integrated_gradients(model, X, steps=32, batch=args.batch)

    # Plots
    bar_plot(imps, feat_names, 'Permutation Importance', os.path.join(args.out_dir, 'explain_permutation.png'))
    bar_plot(sal.tolist(), feat_names, 'Saliency (|∂y/∂x| avg)', os.path.join(args.out_dir, 'explain_saliency.png'))
    bar_plot(ig.tolist(),  feat_names, 'Integrated Gradients (avg |grad|)', os.path.join(args.out_dir, 'explain_integrated_gradients.png'))

    # Summary JSON
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

    print('[Explain] Saved:')
    print(' ', os.path.join(args.out_dir, 'explain_permutation.png'))
    print(' ', os.path.join(args.out_dir, 'explain_saliency.png'))
    print(' ', os.path.join(args.out_dir, 'explain_integrated_gradients.png'))
    print(' ', os.path.join(args.out_dir, 'explain_summary.json'))


if __name__ == '__main__':
    main()
