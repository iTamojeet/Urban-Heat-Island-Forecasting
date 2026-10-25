"""Stage 3: baselines for LST forecasting at 120 m.

Task: predict LST in target month t from information up to month t-1
(last valid observation within K=6 months, calendar month of t).
Models:
  persistence : last valid LST in the window
  climatology : per-cell mean of that calendar month, train years
  linear      : climatology + linear fit on last-valid LST/NDVI anomalies
  linear_era  : linear + concurrent ERA5 anomalies of month t
                (REFERENCE ONLY: uses weather not known at forecast time)
Fit on train years (<=2022), non-held-out cells. Evaluate on 2023 (val) and
2024-25 (test), for all land cells and for held-out spatial blocks.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
K, TRAIN_END, VAL_YEARS, TEST_YEARS = 6, 2022, [2023], [2024, 2025]
FIT_FRAC, SEED = 0.2, 42
MODELS = ["persistence", "climatology", "linear", "linear_era"]

d = np.load(ROOT / "data" / "processed" / "cube.npz")
lst, ndvi, use, keys, era = d["lst"], d["ndvi"], d["usable"], d["dates"], d["era"]
land, test_cell = d["land"], d["test_cell"]
years = np.array([int(k[:4]) for k in keys])
months = np.array([int(k[5:]) for k in keys])
T, H, W = lst.shape
train_t = use & (years <= TRAIN_END)
rng = np.random.default_rng(SEED)


def monthly_clim(x, mask_t):
    out = np.full((13,) + x.shape[1:], np.nan, np.float32)
    for m in range(1, 13):
        idx = np.where(mask_t & (months == m))[0]
        if len(idx):
            out[m] = np.nanmean(x[idx], axis=0)
    return out


clim_l, clim_n = monthly_clim(lst, train_t), monthly_clim(ndvi, train_t)
era_clim = np.stack([era[(years <= TRAIN_END) & (months == m)].mean(0) if m else era[0] * np.nan
                     for m in range(13)])
cell_lst, cell_ndvi = np.nanmean(clim_l[1:], 0), np.nanmean(clim_n[1:], 0)
lst_anom = lst - clim_l[months]
ndvi_anom = ndvi - clim_n[months]
era_anom = era - era_clim[months]


def last_valid(t):
    raw, al, an, gap = (np.full((H, W), np.nan, np.float32) for _ in range(4))
    for lag in range(1, K + 1):
        if t - lag < 0:
            break
        fill = np.isnan(raw) & np.isfinite(lst[t - lag])
        raw[fill], al[fill] = lst[t - lag][fill], lst_anom[t - lag][fill]
        an[fill], gap[fill] = ndvi_anom[t - lag][fill], lag
    return raw, al, an, gap


parts, n_elig, n_dropped = [], 0, 0
for t in range(T):
    if not use[t] or not np.isfinite(clim_l[months[t]]).any():
        continue
    raw, al, an, gap = last_valid(t)
    base = land & np.isfinite(lst[t]) & np.isfinite(clim_l[months[t]])
    ok = base & np.isfinite(raw) & np.isfinite(al) & np.isfinite(an)
    n_elig += base.sum()
    n_dropped += (base & ~ok).sum()
    if years[t] <= TRAIN_END:  # subsample training rows, held-out cells excluded
        ok &= (~test_cell) & (rng.random((H, W)) < FIT_FRAC)
    n = int(ok.sum())
    parts.append(pd.DataFrame({
        "t": t, "year": years[t], "month": months[t], "held_out": test_cell[ok],
        "y": lst[t][ok], "clim": clim_l[months[t]][ok], "persistence": raw[ok],
        "al": al[ok], "an": an[ok], "gap": gap[ok],
        "ndvi_t": ndvi[t][ok], "an_t": ndvi_anom[t][ok],
        "cell_lst": cell_lst[ok], "cell_ndvi": cell_ndvi[ok],
        "e0": era_anom[t, 0], "e1": era_anom[t, 1], "e2": era_anom[t, 2]}, index=range(n)))
df = pd.concat(parts, ignore_index=True)
df["climatology"] = df.clim

fit = df[df.year <= TRAIN_END]
def design(x, era_cols=False):
    cols = [np.ones(len(x)), x.al, x.al * x.gap, x.an]
    if era_cols:
        cols += [x.e0, x.e1, x.e2]
    return np.stack(cols, 1)
beta = np.linalg.lstsq(design(fit), fit.y - fit.clim, rcond=None)[0]
beta_e = np.linalg.lstsq(design(fit, True), fit.y - fit.clim, rcond=None)[0]
df["linear"] = df.clim + design(df) @ beta
df["linear_era"] = df.clim + design(df, True) @ beta_e
df.to_pickle(ROOT / "data" / "processed" / "rows.pkl")
print("linear coefs [1, a_last, a_last*gap, ndvi_a]:", beta.round(3))
print("linear_era coefs (+ era skinT, t2m, precip anomalies):", beta_e.round(3))
print(f"Eligible target pixels dropped (no valid obs in last {K} months): "
      f"{n_dropped / n_elig:.1%}")


def table(sub):
    return {m: dict(rmse=np.sqrt(np.mean((sub[m] - sub.y) ** 2)),
                    mae=np.mean(np.abs(sub[m] - sub.y)), bias=np.mean(sub[m] - sub.y))
            for m in MODELS}


rows = []
for name, yrs in [("val", VAL_YEARS), ("test", TEST_YEARS)]:
    e = df[df.year.isin(yrs)]
    for scope, sub in [("all_land", e), ("held_out_blocks", e[e.held_out])]:
        for m, v in table(sub).items():
            rows.append(dict(split=name, scope=scope, model=m, n=len(sub), **v))
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv(ROOT / "data" / "processed" / "baseline_results.csv", index=False)

te = df[df.year.isin(TEST_YEARS)]
bym = pd.DataFrame({m: te.groupby("month").apply(lambda g: np.sqrt(np.mean((g[m] - g.y) ** 2)))
                    for m in MODELS}).round(2)
print("\nTest RMSE by calendar month:\n", bym.to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
tt = res[(res.split == "test")].pivot(index="model", columns="scope", values="rmse").loc[MODELS]
tt.plot.bar(ax=ax[0], rot=20)
ax[0].set_ylabel("RMSE (degC)")
ax[0].set_title("Test RMSE (2024-25)")
bym.plot(ax=ax[1], marker="o")
ax[1].set_xlabel("calendar month")
ax[1].set_ylabel("RMSE (degC)")
ax[1].set_title("Test RMSE by month")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "baselines.png", dpi=200)