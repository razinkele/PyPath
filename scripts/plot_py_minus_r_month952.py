import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
cmp_fp = BASE / "build" / "seabirds_py_vs_rpath_month952.csv"
out_fp = BASE / "build" / "seabirds_py_minus_r_month952.png"

# load
df = pd.read_csv(cmp_fp)
# pick top 6 by abs_diff (non-null)
df = df.dropna(subset=["abs_diff"]).sort_values("abs_diff", ascending=False).head(6)
# compute diff
df['diff'] = df['Q_py'].astype(float) - df['Q_r'].astype(float)
labels = df['prey_name'].tolist()
diffs = df['diff'].tolist()

x = range(len(labels))

plt.figure(figsize=(8, 4.5))
colors = ['#1f77b4' if d >= 0 else '#d62728' for d in diffs]
plt.bar(x, diffs, color=colors)

for i, d in enumerate(diffs):
    plt.text(i, d + (max(diffs) - min(diffs)) * 0.03, f"{d:.6e}", ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)

plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(x, labels, rotation=30, ha='right')
plt.ylabel('Q_py - Q_r')
plt.title('PyPath minus R reconstructed Q — Month 952 (top diffs)')
plt.tight_layout()
plt.savefig(out_fp, dpi=200)
print(f"Wrote diff plot to {out_fp}")
