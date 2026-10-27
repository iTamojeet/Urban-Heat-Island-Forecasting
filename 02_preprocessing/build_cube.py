"""Stage 2: build the analysis cube at 120 m from monthly Landsat + ERA5.

Steps: keep usable months only -> mask cold outliers (residual cloud) ->
mask NDVI where LST is masked -> 4x4 block-mean to 120 m -> land mask from
median NDVI -> align ERA5 -> per-image NDVI-LST slope (train years) ->
spatial blocks with held-out test blocks. Output: data/processed/cube.npz
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_acquisition"))
import config as C

FACTOR = 4            # 30 m -> 120 m
MIN_VALID_BLOCK = 0.5  # min fraction of valid 30 m pixels per 120 m cell
COLD_OUTLIER = 12.0   # degC below the month's spatial median => masked
MAX_OUTLIER_FRAC = 0.05  # month dropped if more of its valid pixels are outliers
TRAIN_END = 2022
BLOCK = 40            # cells per spatial block side (4.8 km)
TEST_BLOCK_FRAC = 0.2
SEED = 42
WATER_OCC, WATER_FRAC = 25, 0.3  # JRC occurrence >= 25 %; cell water if >= 30 % of its pixels
warnings.filterwarnings("ignore", category=RuntimeWarning)


def block_mean(a, f=FACTOR, min_frac=MIN_VALID_BLOCK):
    y, x = a.shape[0] // f, a.shape[1] // f
    b = a[: y * f, : x * f].reshape(y, f, x, f)
    n = np.sum(~np.isnan(b), axis=(1, 3))
    s = np.nansum(b, axis=(1, 3))
    return np.where(n >= f * f * min_frac, s / np.maximum(n, 1), np.nan).astype(np.float32)


def slope(l, n):
    m = ~np.isnan(l) & ~np.isnan(n)
    if m.sum() < 1000:
        return np.nan, np.nan, int(m.sum())
    return np.polyfit(n[m], l[m], 1)[0], np.corrcoef(n[m], l[m])[0, 1], int(m.sum())


usable = pd.read_csv(C.ROOT / "data" / "interim" / "usable_months.csv")
T = len(usable)
use = usable.usable.values.copy()
dropped = []
keys = [f"{r.year}-{r.month:02d}" for r in usable.itertuples()]

first = next(C.LANDSAT_DIR.glob("landsat_*.tif"))
with rasterio.open(first) as s:
    H, W = s.height // FACTOR, s.width // FACTOR
    full_shape = (s.height, s.width)
    x0, y0 = s.transform.c, s.transform.f

# 30 m water pixels are masked BEFORE aggregation so shoreline cells reflect land only.
occ_path = C.RAW_DIR / "water_occurrence.tif"
water30 = None
if occ_path.exists():
    with rasterio.open(occ_path) as s2:
        water30 = s2.read(1) >= WATER_OCC
    assert water30.shape == full_shape, f"water grid {water30.shape} != {full_shape}"
else:
    print("WARNING: water_occurrence.tif missing; using NDVI rule only")

lst = np.full((T, H, W), np.nan, np.float32)
ndvi = np.full_like(lst, np.nan)
nobs = np.zeros_like(lst)
masked_cnt = {}

for i, r in enumerate(usable.itertuples()):
    if not use[i]:
        continue
    with rasterio.open(C.LANDSAT_DIR / f"landsat_{r.year}_{r.month:02d}.tif") as s:
        a = s.read().astype(np.float32)
    l = np.where(a[0] == C.NODATA, np.nan, a[0] / C.LST_FACTOR)
    n = np.where(a[1] == C.NODATA, np.nan, a[1] / C.NDVI_FACTOR)
    if water30 is not None:
        l[water30] = np.nan
        n[water30] = np.nan
    bad = l < (np.nanmedian(l) - COLD_OUTLIER)
    masked_cnt[keys[i]] = int(bad.sum())
    if bad.sum() / max(np.isfinite(l).sum(), 1) > MAX_OUTLIER_FRAC:
        use[i] = False
        dropped.append(keys[i])
        continue
    l[bad] = np.nan
    n[np.isnan(l)] = np.nan
    lst[i], ndvi[i], nobs[i] = block_mean(l), block_mean(n), block_mean(a[2])

# Land mask: NDVI < 0 rule plus JRC water occurrence (ponds, wetlands, river).
med_ndvi = np.nanmedian(ndvi, axis=0)
land = np.isfinite(med_ndvi) & (med_ndvi >= 0)
n_water_extra = 0
if water30 is not None:
    water_cell = block_mean(water30.astype(np.float32), min_frac=0) >= WATER_FRAC
    n_water_extra = int((land & water_cell).sum())
    land &= ~water_cell
lst[:, ~land], ndvi[:, ~land] = np.nan, np.nan

# ERA5 aligned to the month axis (degC, mm).
era = pd.read_csv(C.ERA5_CSV).set_index("date").reindex(keys)
era_arr = np.stack([era.skin_temperature - 273.15, era.temperature_2m - 273.15,
                    era.total_precipitation_sum * 1000], axis=1).astype(np.float32)

# Per-image spatial NDVI-LST slope on train years.
rows = []
for i, r in enumerate(usable.itertuples()):
    if use[i] and r.year <= TRAIN_END:
        sl, rr, nn = slope(lst[i], ndvi[i])
        rows.append(dict(year=r.year, month=r.month, slope_per_ndvi=sl, r=rr, n=nn))
sl_df = pd.DataFrame(rows)
sl_month = sl_df.groupby("month")[["slope_per_ndvi", "r"]].mean()

# Spatial blocks; hold out a random 20% of land-rich blocks.
by, bx = np.arange(H) // BLOCK, np.arange(W) // BLOCK
block_id = by[:, None] * (bx.max() + 1) + bx[None, :]
ids = [b for b in np.unique(block_id) if land[block_id == b].mean() >= 0.5]
rng = np.random.default_rng(SEED)
test_ids = rng.choice(ids, size=max(1, int(len(ids) * TEST_BLOCK_FRAC)), replace=False)
test_cell = np.isin(block_id, test_ids) & land

out = C.ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "cube.npz", lst=lst, ndvi=ndvi, nobs=nobs,
                    usable=use, dates=np.array(keys), era=era_arr,
                    era_names=np.array(["skin_T_C", "t2m_C", "precip_mm"]), land=land,
                    block_id=block_id, test_cell=test_cell,
                    origin_xy=np.array([x0, y0]), res=FACTOR * C.SCALE)
sl_df.round(4).to_csv(out / "ndvi_lst_slopes_train.csv", index=False)

# Figure: mean Mar-May LST, mean NDVI, mean slope by calendar month.
mo = usable.month.values
spring = use & np.isin(mo, [3, 4, 5])
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a_, d, t, cm in [(ax[0], np.nanmean(lst[spring], 0), "Mean LST, Mar-May (degC)", "inferno"),
                     (ax[1], np.nanmean(ndvi[use], 0), "Mean NDVI", "YlGn")]:
    im = a_.imshow(d, cmap=cm)
    a_.set_title(t)
    a_.axis("off")
    fig.colorbar(im, ax=a_, fraction=0.04)
ax[2].bar(sl_month.index, sl_month.slope_per_ndvi * 0.1)
ax[2].set_xlabel("calendar month")
ax[2].set_ylabel("degC per +0.1 NDVI (spatial, train years)")
ax[2].set_title("Empirical NDVI-LST slope")
fig.tight_layout()
fig.savefig(C.ROOT / "docs" / "figures" / "stage2_overview.png", dpi=200)

print(f"Cube {lst.shape}  usable months {int(use.sum())}  "
      f"land cells {int(land.sum())} ({land.mean():.0%})")
print(f"Extra cells removed by JRC water mask: {n_water_extra}")
print("Dropped for cloud contamination:", dropped)
print("Cold-outlier pixels masked (top 8):",
      sorted(masked_cnt.items(), key=lambda kv: -kv[1])[:8])
print(f"Spatial blocks: {len(ids)} land blocks, {len(test_ids)} held out "
      f"({test_cell.sum()} cells)")
print("Mean NDVI-LST slope, degC per +0.1 NDVI, by month (train years):")
print((sl_month.slope_per_ndvi * 0.1).round(2).to_dict())
print("Mean r by month:", sl_month.r.round(2).to_dict())