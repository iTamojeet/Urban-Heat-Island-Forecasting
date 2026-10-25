"""Stage 1b: monthly ERA5-Land meteorology averaged over the study area.

ERA5-Land is ~9 km, so the study area spans only a handful of cells; the
area mean is used as a dense (gap-free) temporal covariate. Raw units are kept:
temperatures in Kelvin, precipitation in metres of water per month.

Usage:
    python acquire_era5_monthly.py --ee-project YOUR_PROJECT
"""
import argparse
import csv

import ee

import config as C

BANDS = ["skin_temperature", "temperature_2m", "total_precipitation_sum"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-project", default=None)
    args = ap.parse_args()

    C.init_ee(args.ee_project)
    roi = C.get_roi()

    col = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .select(BANDS)
           .filterDate(f"{C.START_YEAR}-01-01", f"{C.END_YEAR + 1}-01-01"))

    def to_feature(img):
        stats = img.reduceRegion(ee.Reducer.mean(), roi, 11132, maxPixels=1e9)
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM"))

    feats = col.map(to_feature).getInfo()["features"]
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.ERA5_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + BANDS)
        for ft in sorted(feats, key=lambda x: x["properties"]["date"]):
            p = ft["properties"]
            w.writerow([p["date"]] + [p.get(b) for b in BANDS])
    print(f"Wrote {len(feats)} rows to {C.ERA5_CSV}")


if __name__ == "__main__":
    main()
