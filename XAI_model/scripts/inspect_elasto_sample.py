import os, glob, numpy as np

base = r"data\elastoplasticity\INPUTS\npy"
samples = sorted(glob.glob(os.path.join(base, "sample_*.npy")))
if not samples:
    raise FileNotFoundError(f"No files like {base}\\sample_*.npy")

f = samples[0]
arr = np.load(f, allow_pickle=True)
print("Sample file:", f)
print("Python type:", type(arr))

# If it’s a dict-like object (common case)
if hasattr(arr, "item"):
    try:
        d = arr.item()
        print("Keys:", list(d.keys()))
        for k in d:
            a = np.array(d[k])
            print(f"  {k}: shape={a.shape}, dtype={a.dtype}, first5={a.flatten()[:5]}")
    except Exception as e:
        print("Not a dict via .item():", e)

# If it’s a plain array
if hasattr(arr, "shape"):
    print("Array shape:", arr.shape, "dtype:", arr.dtype)
    flat = arr.reshape(-1) if arr.size else arr
    print("First 10 values:", flat[:10])
