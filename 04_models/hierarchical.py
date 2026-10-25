"""Stage 4b: two-level LST response model.

LST(c,t) = climatology(c, month) + M(t) + S(c,t)
  M(t): month-level anomaly from ERA5 (linear; only ~55 training months)
  S(c,t): within-month spatial anomaly from NDVI and static cell context
          (millions of pixel rows, so a nonlinear model is justified)
Reports RMSE split exactly into month-level and within-month spatial parts:
total^2 = month^2 + spatial^2.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TRAIN_END, VAL_YEARS, TEST_YEARS = 2022, [2023], [2024, 2025]

df = pd.read_pickle(ROOT / "data" / "processed" / "rows.pkl")
df["anom"] = df.y - df.clim
g = df.groupby(["year", "month"])
df["gm_anom"] = g.anom.transform("mean")
df["nd_c"] = df.ndvi_t - g.ndvi_t.transform("mean")   # centred NDVI (input-only)
df["an_c"] = df.an_t - g.an_t.transform("mean")
df["msin"], df["mcos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)

fit = df[df.year <= TRAIN_END]

# Level 1: month effect from ERA5 anomalies (OLS on month-level means).
mm = fit.groupby(["year", "month"])[["gm_anom", "e0", "e1", "e2"]].first()
X1 = np.column_stack([np.ones(len(mm)), mm[["e0", "e1", "e2"]].values])
b1 = np.linalg.lstsq(X1, mm.gm_anom.values, rcond=None)[0]
print(f"Month model: n_months={len(mm)}  coefs [1, skinT, t2m, precip] = {b1.round(3)}")
df["M"] = np.column_stack([np.ones(len(df)), df[["e0", "e1", "e2"]].values]) @ b1

# Level 2: spatial anomaly.
sp_feats = ["clim", "cell_lst", "cell_ndvi", "ndvi_t", "nd_c", "an_c", "msin", "mcos"]
fit = df[df.year <= TRAIN_END]
s_tr = fit.anom - fit.gm_anom
lin_feats = ["nd_c", "an_c", "cell_ndvi"]
Xl = np.column_stack([np.ones(len(fit))] + [fit[c].fillna(0).values for c in lin_feats])
bl = np.linalg.lstsq(Xl, s_tr.values, rcond=None)[0]
print("Spatial linear coefs [1, nd_c, an_c, cell_ndvi]:", bl.round(3))
df["S_lin"] = np.column_stack([np.ones(len(df))] + [df[c].fillna(0).values for c in lin_feats]) @ bl
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                    min_samples_leaf=500, l2_regularization=5.0,
                                    early_stopping=False, random_state=42)
gbm.fit(fit[sp_feats], s_tr)
df["S_gbm"] = gbm.predict(df[sp_feats])
joblib.dump({"gbm": gbm, "sp_feats": sp_feats, "bl": bl, "lin_feats": lin_feats},
            ROOT / "data" / "processed" / "spatial_model.joblib")

df["hier_lin"] = df.clim + df.M + df.S_lin
df["hier_gbm"] = df.clim + df.M + df.S_gbm
df["month_only"] = df.clim + df.M
MODELS = ["climatology", "linear_era", "month_only", "hier_lin", "hier_gbm"]


def decomp(sub, col):
    err = sub[col] - sub.y
    m = err.groupby([sub.year, sub.month]).transform("mean")
    return np.sqrt((err ** 2).mean()), np.sqrt((m ** 2).mean()), np.sqrt(((err - m) ** 2).mean())


rows = []
for split, yrs in [("val", VAL_YEARS), ("test", TEST_YEARS)]:
    e = df[df.year.isin(yrs)]
    for scope, sub in [("all_land", e), ("held_out_blocks", e[e.held_out])]:
        for model in MODELS:
            t, mo, sp = decomp(sub, model)
            rows.append(dict(split=split, scope=scope, model=model, rmse=t, month_part=mo, spatial_part=sp))
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv(ROOT / "data" / "processed" / "hier_results.csv", index=False)