import os, glob, argparse, json
import numpy as np
import tensorflow as tf

def _from_bytes(b):
    """Decode bytes into numpy array (binary or text)."""
    try:
        arr = np.frombuffer(b, dtype=np.float64)
        if arr.size:
            return arr
    except Exception:
        pass
    try:
        s = b.decode("utf-8")
        arr = np.fromstring(s.replace(",", " "), sep=" ", dtype=np.float64)
        if arr.size:
            return arr
    except Exception:
        pass
    return None

def _extract(ex):
    f = ex.features.feature
    out = {}
    for k in ["strain", "stress", "time", "epsilon", "sigma", "eps", "sig", "t"]:
        if k in f and f[k].bytes_list.value:
            arr = _from_bytes(f[k].bytes_list.value[0])
            if arr is not None:
                out[k] = arr
        elif k in f and f[k].float_list.value:
            out[k] = np.array(f[k].float_list.value, dtype=np.float64)

    if out.get("strain") is None:
        out["strain"] = out.get("epsilon") or out.get("eps")
    if out.get("stress") is None:
        out["stress"] = out.get("sigma") or out.get("sig")
    if out.get("time") is None:
        out["time"] = out.get("t")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to data/viscoelasticity")
    args = ap.parse_args()

    tfr_dir = os.path.join(args.root, "INPUTS", "tfrecord")
    files = sorted(glob.glob(os.path.join(tfr_dir, "sample_*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No TFRecords found in {tfr_dir}")

    strain, stress, times = [], [], []
    for i, path in enumerate(files, 1):
        for rec in tf.data.TFRecordDataset(path):
            ex = tf.train.Example()
            ex.ParseFromString(rec.numpy())
            d = _extract(ex)
            if d.get("strain") is not None and d.get("stress") is not None:
                strain.append(d["strain"].astype(np.float64).reshape(-1,1))
                stress.append(d["stress"].astype(np.float64).reshape(-1,1))
                if d.get("time") is not None:
                    times.append(d["time"].astype(np.float64).reshape(-1,1))
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} TFRecords")

    if not strain or not stress:
        raise RuntimeError("No strain/stress found in TFRecords.")

    X = np.concatenate(strain, axis=0).astype(np.float32)
    Y = np.concatenate(stress, axis=0).astype(np.float32)
    T = np.concatenate(times, axis=0).astype(np.float32) if times else None

    out_strain = os.path.join(args.root, "INPUTS", "npy", "strain.npy")
    out_stress = os.path.join(args.root, "RESULTS", "stress.npy")
    os.makedirs(os.path.dirname(out_strain), exist_ok=True)
    os.makedirs(os.path.dirname(out_stress), exist_ok=True)

    np.save(out_strain, X)
    np.save(out_stress, Y)
    if T is not None:
        np.save(os.path.join(args.root, "INPUTS", "npy", "time.npy"), T)

    meta = {
        "strain_shape": list(X.shape),
        "stress_shape": list(Y.shape),
        "time_shape": list(T.shape) if T is not None else None,
        "source": "tfrecord -> npy converter"
    }
    with open(os.path.join(args.root, "conversion_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" ", out_strain)
    print(" ", out_stress)
    if T is not None:
        print(" ", os.path.join(args.root, "INPUTS", "npy", "time.npy"))
    print("Summary:", meta)

if __name__ == "__main__":
    main()

