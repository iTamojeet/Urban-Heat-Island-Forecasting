# Stage 1: Data Acquisition

Builds the raw inputs for the project: monthly Landsat 8/9 LST and NDVI composites for the study area, a monthly ERA5-Land meteorology table, and a coverage log that shows where the satellite record is thin.

## Concepts

**Land surface temperature (LST)** is the radiometric temperature of the ground and canopy surface as seen by the thermal sensor. It is not air temperature. Urban surfaces such as asphalt and roofs can be many degrees hotter than nearby vegetation under the same air temperature, which is why LST is used to map the surface urban heat island.

**NDVI** is (NIR - Red) / (NIR + Red). For Landsat 8/9 that is band 5 and band 4. Dense green vegetation gives high values, bare and built surfaces give low values, and water gives negative values.

**Collection 2 Level-2** products are already atmospherically corrected. `ST_B10` stores surface temperature as a scaled integer (Kelvin = DN * 0.00341802 + 149.0). `SR_B4` and `SR_B5` store reflectance (reflectance = DN * 0.0000275 - 0.2). `ST_QA` stores per-pixel temperature uncertainty in units of 0.01 K.

**Cloud masking** uses `QA_PIXEL` bits: fill (0), dilated cloud (1), cirrus (2), cloud (3), cloud shadow (4), snow (5). Pixels with `ST_QA` uncertainty above 5 K or LST outside 0-70 degC are also dropped.

**Monthly median composites** reduce the scenes in a month to one value per pixel. The median is robust to residual cloud and haze contamination. `N_OBS` records how many clear scenes contributed, so later stages can down-weight or mask weakly supported months.

**ERA5-Land** is a reanalysis at about 9 km. Over a ~31 x 39 km study area it has only a handful of cells, so it is used as an area-mean temporal covariate, not a spatial one.

## Files

- `config.py`: bbox, CRS, years, scale factors, band order, paths. Edit the bbox here.
- `acquire_landsat_monthly.py`: writes `data/raw/landsat_monthly/landsat_YYYY_MM.tif` and `data/raw/coverage_log.csv`. Resumes if interrupted.
- `acquire_era5_monthly.py`: writes `data/raw/era5_land_monthly.csv`.
- `plot_coverage.py`: writes `docs/figures/coverage_heatmap.png`.

## GeoTIFF layout

Int16, EPSG:32645, 30 m. Band 1 `LST_C_x100` (degC * 100), band 2 `NDVI_x10000`, band 3 `N_OBS`. Nodata is -32768 in bands 1 and 2 and 0 in band 3. Values are stored as scaled integers because `median()` upcasts to float64 and exports balloon in size otherwise.

## Run order

```
pip install -r ../requirements.txt
earthengine authenticate
python acquire_landsat_monthly.py --ee-project YOUR_PROJECT --coverage-only
python plot_coverage.py
python acquire_landsat_monthly.py --ee-project YOUR_PROJECT
python acquire_era5_monthly.py --ee-project YOUR_PROJECT
```

Run the coverage-only pass first and look at the heatmap before the full download. It costs a few minutes and tells you how sparse the monsoon months are.

## Stage gate

Stage 1 is complete when: the coverage heatmap exists, 144 monthly GeoTIFFs (or a documented list of empty months) are on disk, and the ERA5 CSV has 144 rows. Verify that two GeoTIFFs from different months have identical shape and transform before moving on.

## References

- Voogt and Oke (2003), Thermal remote sensing of urban climates, Remote Sensing of Environment.
- USGS Landsat Collection 2 Level-2 Science Product Guide.
- Earth Engine catalog entries: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`, `ECMWF/ERA5_LAND/MONTHLY_AGGR`.
