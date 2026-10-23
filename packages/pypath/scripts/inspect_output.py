"""Inspect RsimOutput attributes."""

import warnings

warnings.filterwarnings("ignore")
from pypath import read_ewemdb, rpath, rsim_scenario, rsim_run

p = read_ewemdb("Data/LT2022_0.5ST_final7.eweaccdb")
m = rpath(p)
s = rsim_scenario(m, p, years=range(1, 4))
out = rsim_run(s, method="AB")
attrs = [a for a in dir(out) if not a.startswith("_")]
print("RsimOutput attrs:", attrs)
for a in attrs:
    v = getattr(out, a)
    if hasattr(v, "shape"):
        print(f"  {a}: shape={v.shape}")
    elif not callable(v):
        print(f"  {a}: {type(v).__name__} = {str(v)[:80]}")
