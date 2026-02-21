import subprocess
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CMP_FP = BASE / "build" / "seabirds_py_vs_rpath_month952.csv"

ABS_TOL = 5e-4
REL_TOL = 2e-3


def _ensure_comparison():
    if not CMP_FP.exists():
        subprocess.run(["python", str(BASE / "scripts" / "compare_py_vs_rpath_month952.py")], check=True)


def test_seabirds_q_diffs_within_tolerance():
    """Fail if reconstructed R-vs-Py Q diffs for month 952 grow beyond tolerances.

    Tolerances are intentionally conservative to catch regressions early.
    """
    _ensure_comparison()
    df = pd.read_csv(CMP_FP)

    # Only consider links where R reconstruction exists
    df_valid = df.dropna(subset=["abs_diff"])
    assert not df_valid.empty, "No valid Q diffs found in comparison CSV"

    max_abs = df_valid["abs_diff"].abs().max()
    max_rel = df_valid["rel_diff"].abs().max()

    # Helpful debug output in case of failure
    if max_abs >= ABS_TOL:
        tops = df_valid.sort_values("abs_diff", ascending=False).head(10)
        raise AssertionError(
            f"Max abs Q diff {max_abs:.3e} exceeded tolerance {ABS_TOL:.0e}\nTop diffs:\n{tops.to_string(index=False)}"
        )

    if max_rel >= REL_TOL:
        tops = df_valid.sort_values("rel_diff", ascending=False).head(10)
        raise AssertionError(
            f"Max rel Q diff {max_rel:.3e} exceeded tolerance {REL_TOL:.0e}\nTop relative diffs:\n{tops.to_string(index=False)}"
        )
