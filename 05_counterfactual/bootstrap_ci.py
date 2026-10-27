"""Stage 5b: month-cluster bootstrap for local vs cross-sectional NDVI response.

Months are the independent units (pixels within a month share weather and
overpass conditions), so whole months are resampled with replacement.
  local : linear spatial model effect of +0.1 local NDVI (nd_c + an_c terms)
  cross : mean per-month cross-sectional slope of LST on NDVI, per +0.1
Uses training-year rows (<= 2022) from rows.pkl.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
B, SEED, TRAIN_END = 2000, 42, 2022
SEASONS = {"all": range(1, 13), "Jan-Feb": [1, 2], "Mar-May": [3, 4, 5], "Oct-Dec": [10, 11, 12]}

df = pd.read_pickle(ROOT / "data" / "processed" / "rows.pkl")
df = df[df.year <= TRAIN_END].copy()
df["anom"] = df.y - df.clim
g = df.groupby(["year", "month"])
df["s"] = df.anom - g.anom.transform("mean")
df["nd_c"] = df.ndvi_t - g.ndvi_t.transform("mean")
df["an_c"] = df.an_t - g.an_t.transform("mean")
df = df.dropna(subset=["ndvi_t", "an_t", "nd_c", "an_c", "cell_ndvi", "s"])

stats = []
for (yr, mo), x in df.groupby(["year", "month"]):
    X = np.column_stack([np.ones(len(x)), x.nd_c, x.an_c, x.cell_ndvi])
    stats.append(dict(month=mo, XtX=X.T @ X, Xty=X.T @ x.s.values,
                      xs=np.cov(x.ndvi_t, x.y)[0, 1] / np.var(x.ndvi_t, ddof=1)))


def estimate(sel):
    b = np.linalg.solve(sum(s["XtX"] for s in sel), sum(s["Xty"] for s in sel))
    return 0.1 * (b[1] + b[2]), 0.1 * np.mean([s["xs"] for s in sel])


rng = np.random.default_rng(SEED)
rows = []
for name, months in SEASONS.items():
    sub = [s for s in stats if s["month"] in months]
    if len(sub) < 3:
        continue
    loc, cross = estimate(sub)
    boots = np.array([estimate([sub[i] for i in rng.integers(0, len(sub), len(sub))])
                      for _ in range(B)])
    ratio = boots[:, 0] / boots[:, 1]
    q = lambda a: np.percentile(a, [2.5, 97.5])
    rows.append(dict(season=name, n_months=len(sub),
                     local=loc, local_lo=q(boots[:, 0])[0], local_hi=q(boots[:, 0])[1],
                     cross=cross, cross_lo=q(boots[:, 1])[0], cross_hi=q(boots[:, 1])[1],
                     ratio=loc / cross, ratio_lo=q(ratio)[0], ratio_hi=q(ratio)[1]))
out = pd.DataFrame(rows).round(2)
print("degC per +0.1 NDVI, 95% month-bootstrap intervals")
print(out.to_string(index=False))
out.to_csv(ROOT / "data" / "processed" / "ndvi_response_ci.csv", index=False)
