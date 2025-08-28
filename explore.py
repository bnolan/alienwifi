import h5py
import numpy as np

with h5py.File("data/taucet.h5", "r") as f:
    print(list(f.attrs.keys()))

with h5py.File("data/taucet.h5", "r") as f:
    fch1 = f.attrs.get("fch1")
    foff = f.attrs.get("foff")
    nchans = f["data"].shape[2]

    freqs = fch1 + np.arange(nchans) * foff
    print(freqs[:10])
    print(freqs[-10:])