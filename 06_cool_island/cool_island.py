"""Stage 6: cooling reach of the Hooghly river and other water bodies.

For each usable month, LST is regressed on NDVI across land cells; the residual is
the LST not explained by vegetation. Residuals are binned by distance to the nearest
water (river = largest JRC water component, everything else = "other water") and
expressed relative to cells farther than 2.88 km from any water. Profiles are
averaged over months in a season, with month-cluster bootstrap intervals.
Caveat: the river runs through the dense core, so impervious cover is not controlled.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
F, OCC, B, SEED = 4, 25, 2000, 42
EDGES = np.array([0, 120, 240, 360, 480, 720, 960, 1440, 1920, 2880])
MIDS = (EDGES[:-1] + EDGES[1:]) / 2
SEASONS = {"Jan-Feb": [1, 2], "Mar-May": [3, 4, 5], "Oct-Dec": [10, 11, 12]}
TYPES = ["Hooghly river", "other water"]

d = np.load(ROOT / "data" / "processed" / "cube.npz")
lst, ndvi, use, keys, land = d["lst"], d["ndvi"], d["usable"], d["dates"], d["land"]
months = np.array([int(k[5:]) for k in keys])
H, W = land.shape

with rasterio.open(ROOT / "data" / "raw" / "water_occurrence.tif") as s:
    water = s.read(1) >= OCC
lab, n = ndimage.label(water, structure=np.ones((3, 3)))
sizes = ndimage.sum(water, lab, index=np.arange(1, n + 1))
order = np.argsort(sizes)[::-1]
print("Largest water components (km2):", (sizes[order[:5]] * 900 / 1e6).round(2))
river = lab == (order[0] + 1)
other = water & ~river


def to_cell(a):
    return a[: H * F, : W * F].reshape(H, F, W, F).mean(axis=(1, 3))


dr = to_cell(ndimage.distance_transform_edt(~river, sampling=30))
do = to_cell(ndimage.distance_transform_edt(~other, sampling=30)) if other.any() else np.full((H, W), np.inf)
dist, is_river = np.minimum(dr, do), dr <= do
bin_idx = np.digitize(dist, EDGES) - 1  # 0..8 = bins, 9 = far field
far = dist >= EDGES[-1]
if (land & far).sum() < 200:  # small domain: use the farthest 10% of land as reference
    far = dist >= np.percentile(dist[land], 90)
    print("Far-field reference: farthest 10% of land cells")
sel_type = {"Hooghly river": is_river, "other water": ~is_river}
n_cells = {t: np.array([(land & sel_type[t] & (bin_idx == k)).sum() for k in range(len(MIDS))]) for t in TYPES}


def month_profile(t):
    l, nd = lst[t], ndvi[t]
    ok = land & np.isfinite(l) & np.isfinite(nd)
    slope, icpt = np.polyfit(nd[ok], l[ok], 1)
    res = l - (icpt + slope * nd)
    ref = res[ok & far].mean() if (ok & far).sum() > 50 else np.nan
    out = np.full((2, len(MIDS)), np.nan)
    for i, tname in enumerate(TYPES):
        for k in range(len(MIDS)):
            m = ok & sel_type[tname] & (bin_idx == k)
            if m.sum() >= 20:
                out[i, k] = res[m].mean() - ref
    return out


rng = np.random.default_rng(SEED)
rows = []
for season, mos in SEASONS.items():
    idx = [t for t in range(len(keys)) if use[t] and months[t] in mos]
    if len(idx) < 3:
        continue
    prof = np.stack([month_profile(t) for t in idx])  # (months, 2, bins)
    mean = np.nanmean(prof, axis=0)
    boots = np.stack([np.nanmean(prof[rng.integers(0, len(idx), len(idx))], axis=0) for _ in range(B)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    for i, tname in enumerate(TYPES):
        for k in range(len(MIDS)):
            rows.append(dict(season=season, water=tname, dist_mid_m=MIDS[k], n_months=len(idx),
                             n_cells=int(n_cells[tname][k]), dLST=mean[i, k], lo=lo[i, k], hi=hi[i, k]))
out = pd.DataFrame(rows).round(2)
out.to_csv(ROOT / "data" / "processed" / "cool_island_profile.csv", index=False)
print("\nMar-May cooling vs cells > 2.9 km from water (degC):")
print(out[out.season == "Mar-May"].pivot(index="dist_mid_m", columns="water", values="dLST").to_string())

fig, axes = plt.subplots(1, len(SEASONS), figsize=(14, 4.2), sharey=True)
for ax, season in zip(axes, SEASONS):
    for tname, c in zip(TYPES, ["tab:blue", "tab:green"]):
        s_ = out[(out.season == season) & (out.water == tname)]
        ax.plot(s_.dist_mid_m, s_.dLST, marker="o", color=c, label=tname)
        ax.fill_between(s_.dist_mid_m, s_.lo, s_.hi, color=c, alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{season} ({int(out[out.season == season].n_months.iloc[0])} months)")
    ax.set_xlabel("distance to nearest water (m)")
axes[0].set_ylabel("LST relative to cells > 2.9 km from water (degC)\n(vegetation-adjusted)")
axes[0].legend()
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "cool_island.png", dpi=200)
