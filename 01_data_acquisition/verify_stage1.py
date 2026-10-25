"""Stage 1 gate: verify downloads, check ERA5, flag usable months.

Writes data/interim/usable_months.csv. A month is usable when >= MIN_VALID
of pixels have a clear LST observation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_acquisition"))
import config as C

MIN_VALID = 0.80
cov = pd.read_csv(C.COVERAGE_CSV)
ref, problems, rows = None, [], []

for r in cov.itertuples():
    f = C.LANDSAT_DIR / f"landsat_{r.year}_{r.month:02d}.tif"
    d = dict(year=r.year, month=r.month, has_file=f.exists(), valid_frac=0.0,
             lst_p1=np.nan, lst_p99=np.nan, ndvi_p1=np.nan, ndvi_p99=np.nan)
    if f.exists():
        with rasterio.open(f) as src:
            sig = (src.count, src.height, src.width, str(src.crs),
                   tuple(round(v, 3) for v in src.transform)[:6], src.dtypes)
            if ref is None:
                ref = sig
                print("Reference grid:", sig)
            elif sig != ref:
                problems.append(f"{f.name}: grid differs from reference")
            lst, ndvi = src.read(1), src.read(2)
        m = lst != C.NODATA
        d["valid_frac"] = round(float(m.mean()), 3)
        if m.any():
            d["lst_p1"], d["lst_p99"] = np.percentile(lst[m] / C.LST_FACTOR, [1, 99])
            d["ndvi_p1"], d["ndvi_p99"] = np.percentile(ndvi[m] / C.NDVI_FACTOR, [1, 99])
            if d["lst_p1"] < 5 or d["lst_p99"] > 60:
                problems.append(f"{f.name}: LST range {d['lst_p1']:.1f}-{d['lst_p99']:.1f} degC")
    rows.append(d)

df = pd.DataFrame(rows)
df["usable"] = df["valid_frac"] >= MIN_VALID
out = C.ROOT / "data" / "interim" / "usable_months.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.round(2).to_csv(out, index=False)

print(f"\nFiles: {int(df.has_file.sum())}/{len(df)}  usable (>= {MIN_VALID:.0%} valid): {int(df.usable.sum())}")
print("Missing files:", ", ".join(f"{y}-{m:02d}" for y, m in df.loc[~df.has_file, ["year", "month"]].values) or "none")
print("Usable months per calendar month:", df.groupby("month").usable.sum().to_dict())
print("LST p1/p99 over usable months (degC): "
      f"{df.loc[df.usable, 'lst_p1'].min():.1f} / {df.loc[df.usable, 'lst_p99'].max():.1f}")

era = pd.read_csv(C.ERA5_CSV)
print(f"\nERA5 rows: {len(era)}  NaNs: {int(era.isna().sum().sum())}  "
      f"skin T range: {era.skin_temperature.min() - 273.15:.1f}-{era.skin_temperature.max() - 273.15:.1f} degC")
print("\nProblems:", problems or "none")
