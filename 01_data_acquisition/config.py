"""Shared configuration for Stage 1 (data acquisition)."""
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LANDSAT_DIR = RAW_DIR / "landsat_monthly"
ERA5_CSV = RAW_DIR / "era5_land_monthly.csv"
COVERAGE_CSV = RAW_DIR / "coverage_log.csv"

# Study area: Kolkata metropolitan core (lon_min, lat_min, lon_max, lat_max).
# Roughly 31 km x 39 km. Confirm against a basemap before the full download.
BBOX = (88.20, 22.40, 88.50, 22.75)

# UTM zone 45N covers 84E-90E, which contains Kolkata.
CRS = "EPSG:32645"
SCALE = 30  # metres; native Landsat resolution

# Landsat 8 starts Apr 2013 (first full year 2014); Landsat 9 starts Nov 2021.
START_YEAR = 2014
END_YEAR = 2025

# Level-2 scale factors (USGS Collection 2 Level-2 Science Product Guide).
ST_SCALE, ST_OFFSET = 0.00341802, 149.0  # ST_B10 DN -> Kelvin
SR_SCALE, SR_OFFSET = 0.0000275, -0.2    # SR_B* DN -> reflectance
ST_QA_MAX = 500  # ST_QA is in 0.01 K units; mask where uncertainty > 5 K

# Integer storage (learnings: median() upcasts to float64; cast back to int16).
LST_FACTOR = 100      # degC * 100
NDVI_FACTOR = 10000   # NDVI * 10000
NODATA = -32768

# Band order in each exported GeoTIFF.
BANDS = ["LST_C_x100", "NDVI_x10000", "N_OBS"]


def get_roi() -> ee.Geometry:
    return ee.Geometry.Rectangle(list(BBOX), geodesic=False)


def init_ee(project: str | None = None) -> None:
    ee.Initialize(project=project) if project else ee.Initialize()
