"""Final figure: Mar-May local NDVI scenarios with month-bootstrap intervals."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sc = pd.read_csv(PROC / "ndvi_scenarios.csv")
ci = pd.read_csv(PROC / "ndvi_response_ci.csv").set_index("season").loc["Mar-May"]
m = sc[sc.group == "Mar-May"].reset_index(drop=True)
k = m.dNDVI.values / 0.1  # scale per-0.1 estimates to each scenario's mean dNDVI

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x, w = np.arange(len(m)), 0.27
ax.bar(x - w, m.gbm, w, label="Nonlinear model (test-year cells)")
for off, name, v, lo, hi in [(0, "Local linear response (95% bootstrap)", ci.local, ci.local_lo, ci.local_hi),
                             (w, "Cross-sectional slope (95% bootstrap)", ci.cross, ci.cross_lo, ci.cross_hi)]:
    val = v * k
    ax.bar(x + off, val, w, label=name,
           yerr=[val - lo * k, hi * k - val], capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x, [f"{s} NDVI" for s in m.scenario])
ax.set_ylabel("change in LST (degC)")
ax.set_title("Mar-May: simulated local greening, Kolkata")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
out = ROOT / "docs" / "figures" / "marmay_scenarios.png"
fig.savefig(out, dpi=200)
print("Wrote", out)
