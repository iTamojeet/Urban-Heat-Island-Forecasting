"""Stage 4a: gradient-boosted trees on the Stage 3 rows.

gbm_prev : inputs known through month t-1 (true forecasting information)
gbm_conc : adds concurrent NDVI and ERA5 anomalies of month t
           (emulator: LST given weather + vegetation state; used for NDVI scenarios)
Target is the anomaly (y - climatology). Heavy regularisation because the ERA5
features take only ~60 distinct values in training (one per month).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TRAIN_END, VAL_YEARS, TEST_YEARS = 2022, [2023], [2024, 2025]

df = pd.read_pickle(ROOT / "data" / "processed" / "rows.pkl")
df["msin"], df["mcos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)
prev = ["clim", "cell_lst", "cell_ndvi", "msin", "mcos", "al", "an", "gap"]
conc = prev + ["ndvi_t", "an_t", "e0", "e1", "e2"]
fit = df[df.year <= TRAIN_END]

for name, feats in [("gbm_prev", prev), ("gbm_conc", conc)]:
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=5000, l2_regularization=10.0,
                                      early_stopping=False, random_state=42)
    m.fit(fit[feats], fit.y - fit.clim)
    df[name] = df.clim + m.predict(df[feats])

MODELS = ["climatology", "linear", "linear_era", "gbm_prev", "gbm_conc"]
rows = []
for split, yrs in [("val", VAL_YEARS), ("test", TEST_YEARS)]:
    e = df[df.year.isin(yrs)]
    for scope, sub in [("all_land", e), ("held_out_blocks", e[e.held_out])]:
        for mm in MODELS:
            err = sub[mm] - sub.y
            rows.append(dict(split=split, scope=scope, model=mm, n=len(sub),
                             rmse=np.sqrt((err ** 2).mean()), mae=err.abs().mean(), bias=err.mean()))
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv(ROOT / "data" / "processed" / "gbm_results.csv", index=False)
te = df[df.year.isin(TEST_YEARS)]
print("\nTest RMSE by calendar month:\n", pd.DataFrame(
    {mm: te.groupby("month").apply(lambda g: np.sqrt(((g[mm] - g.y) ** 2).mean()))
     for mm in MODELS}).round(2).to_string())
