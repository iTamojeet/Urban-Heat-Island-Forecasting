"""Stage 6: local NDVI scenarios on test-year rows (2024-25).

Scenario: NDVI of a cell rises by p% of its own value while the city-wide
monthly mean is held fixed (local greening). Only the within-month spatial
model responds; month-level weather is unchanged. Three estimates of the LST change:
  gbm            spatial GBM from hierarchical.py
  linear_local   linear spatial model (nd_c + an_c terms)
  cross_section  empirical cross-sectional slope (train years) x dNDVI
Cross-sectional slopes confound NDVI with urban form, so they are expected
to overstate cooling relative to the model's local response.
"""
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEST_YEARS, SCENARIOS = [2024, 2025], (0.1, 0.2, 0.3)

df = pd.read_pickle(PROC / "rows.pkl")
g = df.groupby(["year", "month"])
df["nd_c"] = df.ndvi_t - g.ndvi_t.transform("mean")
df["an_c"] = df.an_t - g.an_t.transform("mean")
df["msin"], df["mcos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)

b = joblib.load(PROC / "spatial_model.joblib")
gbm, feats, bl, lin_feats = b["gbm"], b["sp_feats"], b["bl"], b["lin_feats"]
slopes = pd.read_csv(PROC / "ndvi_lst_slopes_train.csv").groupby("month").slope_per_ndvi.mean()

te = df[df.year.isin(TEST_YEARS) & df.ndvi_t.notna() & df.an_t.notna()].copy()
te["xsl"] = te.month.map(slopes)
te = te[te.xsl.notna()]
te["grp"] = pd.qcut(te.cell_ndvi, 3, labels=["low NDVI", "mid NDVI", "high NDVI"])


def lin(x):
    X = np.column_stack([np.ones(len(x))] + [x[c].fillna(0).values for c in lin_feats])
    return X @ bl


base_g, base_l = gbm.predict(te[feats]), lin(te)
rows = []
for p in SCENARIOS:
    d = te.ndvi_t * p
    s = te.copy()
    s["ndvi_t"], s["nd_c"], s["an_c"] = te.ndvi_t + d, te.nd_c + d, te.an_c + d
    res = pd.DataFrame({"grp": te.grp.values, "dNDVI": d.values,
                        "gbm": gbm.predict(s[feats]) - base_g,
                        "linear_local": lin(s) - base_l,
                        "cross_section": (te.xsl * d).values})
    res["gbm_cools"] = res.gbm < 0
    for name, sub in [("all", res)] + list(res.groupby("grp")):
        rows.append(dict(scenario=f"+{int(p * 100)}%", group=name, dNDVI=sub.dNDVI.mean(),
                         gbm=sub.gbm.mean(), linear_local=sub.linear_local.mean(),
                         cross_section=sub.cross_section.mean(),
                         pct_cells_gbm_cools=100 * sub.gbm_cools.mean()))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv(PROC / "ndvi_scenarios.csv", index=False)

a = out[out.group == "all"].set_index("scenario")[["gbm", "linear_local", "cross_section"]]
ax = a.plot.bar(rot=0, figsize=(7, 4.5))
ax.set_ylabel("mean change in LST (degC)")
ax.set_title("Local NDVI scenarios, test years: three estimates")
plt.tight_layout()
plt.savefig(ROOT / "docs" / "figures" / "ndvi_scenarios.png", dpi=200)
