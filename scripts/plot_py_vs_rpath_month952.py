from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
cmp_fp = BASE / "build" / "seabirds_py_vs_rpath_month952.csv"
out_fp = BASE / "build" / "seabirds_py_vs_rpath_month952.png"

df = pd.read_csv(cmp_fp)
# pick top 6 by abs_diff (non-null)
df = df.dropna(subset=["abs_diff"]).sort_values("abs_diff", ascending=False).head(6)
labels = df["prey_name"].tolist()
Q_py = df["Q_py"].astype(float).tolist()
Q_r = df["Q_r"].astype(float).tolist()
abs_diff = df["abs_diff"].astype(float).tolist()

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar([i - width/2 for i in x], Q_py, width=width, label='PyPath Q_final', color='#1f77b4')
plt.bar([i + width/2 for i in x], Q_r, width=width, label='R reconstructed Q', color='#ff7f0e')

for i, (a, b) in enumerate(zip(Q_py, Q_r)):
    plt.text(i - width/2, a + max(Q_py+Q_r) * 0.01, f"{a:.6f}", ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, b + max(Q_py+Q_r) * 0.01, f"{b:.6f}", ha='center', va='bottom', fontsize=8)

plt.xticks(x, labels, rotation=30, ha='right')
plt.ylabel('Q (consumption)')
plt.title('PyPath vs R reconstructed Q — Month 952 (top diffs)')
plt.legend()
plt.tight_layout()
plt.savefig(out_fp, dpi=200)
print(f"Wrote plot to {out_fp}")
