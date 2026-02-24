import h5py
import sys

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name:60s} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"[GROUP]   {name}")

if __name__ == "__main__":
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        f.visititems(walk)
