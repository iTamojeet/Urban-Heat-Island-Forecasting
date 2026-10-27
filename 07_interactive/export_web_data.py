"""Stage 7: export compact data for the interactive map (docs/map/data.js).

Mar-May mean LST and NDVI over usable months, quantised to uint8 and base64 encoded,
plus the local and cross-sectional response estimates from bootstrap_ci.py.
"""
import base64
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.warp import transform

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MONTHS, MIN_MONTHS = [3, 4, 5], 5

d = np.load(ROOT / "data" / "processed" / "cube.npz")
lst, ndvi, use, keys = d["lst"], d["ndvi"], d["usable"], d["dates"]
months = np.array([int(k[5:]) for k in keys])
sel = [t for t in range(len(keys)) if use[t] and months[t] in MONTHS]
ml, mn = np.nanmean(lst[sel], axis=0), np.nanmean(ndvi[sel], axis=0)
cnt = np.isfinite(lst[sel]).sum(axis=0)
land = d["land"] & (cnt >= MIN_MONTHS) & np.isfinite(ml) & np.isfinite(mn)
H, W = ml.shape

ql = np.clip(np.round((ml - 25) * 10), 0, 254).astype(np.uint8)
qn = np.clip(np.round(np.nan_to_num(mn) * 250), 0, 254).astype(np.uint8)
ql[~land], qn[~land] = 255, 255

x0, y0 = [float(v) for v in d["origin_xy"]]
res = float(d["res"])
xs, ys = [x0, x0 + W * res, x0, x0 + W * res], [y0, y0, y0 - H * res, y0 - H * res]
lons, lats = transform("EPSG:32645", "EPSG:4326", xs, ys)
corners = {k: [round(la, 6), round(lo, 6)] for k, la, lo in zip(["TL", "TR", "BL", "BR"], lats, lons)}

ci = pd.read_csv(ROOT / "data" / "processed" / "ndvi_response_ci.csv").set_index("season").loc["Mar-May"]
out = dict(
    H=H, W=W, res_m=res, n_months=len(sel), corners=corners,
    lst=base64.b64encode(ql.tobytes()).decode(), ndvi=base64.b64encode(qn.tobytes()).decode(),
    lst_min=25, lst_scale=10, nd_scale=250,
    ndvi_cap=round(float(np.percentile(mn[land], 99)), 3),
    land_cells=int(land.sum()), mean_lst=round(float(np.mean(ml[land])), 2),
    response=dict(local=float(ci.local), local_lo=float(ci.local_lo), local_hi=float(ci.local_hi),
                  cross=float(ci.cross), cross_lo=float(ci.cross_lo), cross_hi=float(ci.cross_hi),
                  n_months=int(ci.n_months)),
)
path = ROOT / "docs" / "map" / "data.js"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("window.KOLKATA = " + json.dumps(out) + ";\n")
print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB): {H}x{W} cells, {int(land.sum())} land, "
      f"{len(sel)} Mar-May months, ndvi cap {out['ndvi_cap']}")
