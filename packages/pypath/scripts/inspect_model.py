"""Quick inspection of balanced model and scenario attributes."""

import warnings

from pypath import read_ewemdb, rpath, rsim_run, rsim_scenario

warnings.filterwarnings("ignore")

p = read_ewemdb("Data/LT2022_0.5ST_final7.eweaccdb")
m = rpath(p)

print("=== Rpath (m) ===")
print("Biomass:", m.Biomass[:5])
print("Group:", m.Group[:5])
print("PB:", m.PB[:5])
print("QB:", m.QB[:5])
print("EE:", m.EE[:5])
print("type:", m.type[:5])

s = rsim_scenario(m, p, years=range(1, 4))
print("\n=== RsimParams (s.params) ===")
attrs = [a for a in dir(s.params) if not a.startswith("_")]
print("All attrs:", attrs)
print("BB:", s.params.BB[:5] if hasattr(s.params, "BB") else "N/A")
print("VV shape:", s.params.VV.shape if hasattr(s.params, "VV") else "N/A")

print("\n=== Quick Ecosim run ===")
try:
    out = rsim_run(s, method="AB")
    print("biomass shape:", out.biomass.shape)
    print(
        "annual_Biomass shape:",
        out.annual_Biomass.shape if hasattr(out, "annual_Biomass") else "N/A",
    )
    print(
        "annual_biomass shape:",
        out.annual_biomass.shape if hasattr(out, "annual_biomass") else "N/A",
    )
    outattrs = [a for a in dir(out) if not a.startswith("_")]
    print("RsimOutput attrs:", outattrs)
except Exception as e:
    print(f"Run failed: {e}")
    import traceback

    traceback.print_exc()
