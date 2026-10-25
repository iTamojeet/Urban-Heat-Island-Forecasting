"""Stage 1a: monthly Landsat 8/9 LST + NDVI composites for the study area.

For every month it (1) logs scene count and clear-observation coverage to
coverage_log.csv and (2) downloads a 3-band int16 GeoTIFF:
    band 1  LST_C_x100    median land surface temperature, degC * 100
    band 2  NDVI_x10000   median NDVI * 10000
    band 3  N_OBS         clear LST observations per pixel in that month
Missing pixels are -32768 in bands 1-2 and 0 in band 3.

Usage:
    python acquire_landsat_monthly.py --ee-project YOUR_PROJECT --coverage-only
    python acquire_landsat_monthly.py --ee-project YOUR_PROJECT
"""
import argparse
import csv

import ee
import geemap

import config as C


def preprocess(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    # QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow
    bad = ee.Image(0)
    for bit in (0, 1, 2, 3, 4, 5):
        bad = bad.Or(qa.bitwiseAnd(1 << bit).neq(0))
    bad = bad.Or(img.select("ST_QA").gt(C.ST_QA_MAX))

    lst = (img.select("ST_B10").multiply(C.ST_SCALE).add(C.ST_OFFSET)
           .subtract(273.15).rename("LST_C"))
    nir = img.select("SR_B5").multiply(C.SR_SCALE).add(C.SR_OFFSET)
    red = img.select("SR_B4").multiply(C.SR_SCALE).add(C.SR_OFFSET)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    valid = bad.Not().And(lst.gt(0)).And(lst.lt(70))  # drop fill / implausible
    return (lst.addBands(ndvi).updateMask(valid)
            .copyProperties(img, ["system:time_start"]))


def month_collection(year: int, month: int, roi: ee.Geometry) -> ee.ImageCollection:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    return l8.merge(l9).filterBounds(roi).filterDate(start, end)


def build_composite(col: ee.ImageCollection) -> ee.Image:
    proc = col.map(preprocess)
    med = proc.median()
    lst = med.select("LST_C").multiply(C.LST_FACTOR).round().unmask(C.NODATA).toInt16()
    ndvi = med.select("NDVI").multiply(C.NDVI_FACTOR).round().unmask(C.NODATA).toInt16()
    nobs = proc.select("LST_C").count().unmask(0).toInt16().rename("N_OBS")
    return ee.Image.cat([lst.rename(C.BANDS[0]), ndvi.rename(C.BANDS[1]), nobs])


def coverage_row(year: int, month: int, col: ee.ImageCollection, roi: ee.Geometry) -> dict:
    n_scenes = col.size().getInfo()
    row = {"year": year, "month": month, "n_scenes": n_scenes,
           "mean_clear_obs": 0.0, "frac_zero_obs": 1.0}
    if n_scenes == 0:
        return row
    nobs = col.map(preprocess).select("LST_C").count().unmask(0)
    stats = (ee.Image.cat([nobs.rename("m"), nobs.eq(0).rename("z")])
             .reduceRegion(ee.Reducer.mean(), roi, 120, maxPixels=1e9).getInfo())
    row["mean_clear_obs"] = round(stats["m"], 3)
    row["frac_zero_obs"] = round(stats["z"], 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-project", default=None)
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--start-year", type=int, default=C.START_YEAR)
    ap.add_argument("--end-year", type=int, default=C.END_YEAR)
    args = ap.parse_args()

    C.init_ee(args.ee_project)
    roi = C.get_roi()
    C.LANDSAT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            col = month_collection(year, month, roi)
            row = coverage_row(year, month, col, roi)
            rows.append(row)
            print(f"{year}-{month:02d}  scenes={row['n_scenes']:3d}  "
                  f"clear_obs={row['mean_clear_obs']:.2f}  zero={row['frac_zero_obs']:.2f}")
            if args.coverage_only or row["n_scenes"] == 0:
                continue
            out = C.LANDSAT_DIR / f"landsat_{year}_{month:02d}.tif"
            if out.exists():
                continue  # resume support
            geemap.download_ee_image(build_composite(col), str(out),
                                     region=roi, crs=C.CRS, scale=C.SCALE)

    with open(C.COVERAGE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "month", "n_scenes",
                                          "mean_clear_obs", "frac_zero_obs"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {C.COVERAGE_CSV}")


if __name__ == "__main__":
    main()
