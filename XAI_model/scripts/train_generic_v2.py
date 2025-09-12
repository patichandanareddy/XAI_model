# scripts/train_generic_v2.py
import argparse, os, glob, json
from typing import List, Dict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PREFERRED_FEATURES = ["strain","prev_strain","prev_stress","e_visco","prev_e_visco","control","time_step","rate"]
PREFERRED_PARAMS   = ["E","eta","H_kin","sig0"]

def load_one(path):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        return d.item()
    if isinstance(d, np.lib.npyio.NpzFile):
        return dict(d.items())
    return d

def pick_feats(d: dict, use_params: bool):
    keys = [k for k in PREFERRED_FEATURES if k in d]
    if use_params: keys += [k for k in PREFERRED_PARAMS if k in d]
    if not keys:
        for k,v in d.items():
            if k.lower() in ("stress","delta_stress","target"): continue
            v = np.asarray(v)
            if v.ndim==1 and np.issubdtype(v.dtype, np.number): keys.append(k)
    return keys

def build_xy(d: dict, feats: List[str], target_delta: bool):
    N = len(d[feats[0]]) if feats else len(d["strain"])
    X = np.concatenate([np.asarray(d[k]).reshape(-1,1).astype(np.float32) if np.ndim(d[k])>0
                        else np.full((N,1), float(d[k]), np.float32) for k in feats], axis=1)
    if "stress" in d: y = np.asarray(d["stress"], np.float32).reshape(-1,1)
    elif "target" in d: y = np.asarray(d["target"], np.float32).reshape(-1,1)
    else: raise KeyError("No 'stress' or 'target' in data")
    if target_delta:
        if "prev_stress" in d:
            y = y - np.asarray(d["prev_stress"], np.float32).reshape(-1,1)
        elif "delta_stress" in d:
            y = np.asarray(d["delta_stress"], np.float32).reshape(-1,1)
    return X, y

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256,256,128,64], out_dim=1):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden:
            layers += [nn.Linear(cur,h), nn.ReLU()]
            cur = h
        layers += [nn.Linear(cur,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/viscoelastic_v2")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--use_params", action="store_true")
    ap.add_argument("--target_delta", action="store_true")
    ap.add_argument("--max_files", type=int, default=500)
    ap.add_argument("--limit", type=int, default=400000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "INPUTS", "npy", "sample_*.npy")))
    if args.max_files: files = files[:args.max_files]

    Xs, Ys, feat_names, total = [], [], None, 0
    for f in tqdm(files, desc="[Load]"):
        d = load_one(f)
        feats = pick_feats(d, args.use_params)
        if not feats: continue
        X,y = build_xy(d, feats, args.target_delta)
        if feat_names is None: feat_names = feats
        Xs.append(X); Ys.append(y)
        total += len(X)
        if args.limit and total >= args.limit: break

    X = np.concatenate(Xs,0); y = np.concatenate(Ys,0).reshape(-1)
    stats = {}
    if args.standardize:
        mean = X.mean(0, keepdims=True); std = X.std(0, keepdims=True)+1e-8
        X = (X-mean)/std
        stats["x_mean"]=mean.tolist(); stats["x_std"]=std.tolist()

    X_t = torch.from_numpy(X); y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model = MLP(in_dim=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    for ep in range(1, args.epochs+1):
        model.train()
        losses=[]
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep}/{args.epochs}  TrainLoss: {np.mean(losses):.6e}")

    torch.save(model.state_dict(), os.path.join(args.out_dir,"model.pt"))
    with open(os.path.join(args.out_dir,"summary.json"),"w") as f:
        json.dump({"in_dim": int(X.shape[1]), "features": feat_names, **stats}, f, indent=2)
    print("[Done] Saved:", os.path.join(args.out_dir,"model.pt"))

if __name__ == "__main__":
    main()
