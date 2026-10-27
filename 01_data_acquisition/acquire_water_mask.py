"""Stage 1d: static JRC Global Surface Water occurrence (0-100 %) on the Landsat grid.

Used in Stage 2 to mask rivers, ponds and wetlands that the NDVI < 0 rule misses.
Usage: python acquire_water_mask.py --ee-project YOUR_PROJECT
"""
import argparse

import ee
import geemap

import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--ee-project", default=None)
args = ap.parse_args()
C.init_ee(args.ee_project)
roi = C.get_roi()
occ = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
       .unmask(0).toInt16().rename("occurrence"))
out = C.RAW_DIR / "water_occurrence.tif"
geemap.download_ee_image(occ, str(out), region=roi, crs=C.CRS, scale=C.SCALE)
print("Wrote", out)
