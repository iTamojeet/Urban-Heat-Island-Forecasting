# Glossary

**Baseline**: a simple reference forecast (persistence, seasonal climatology, linear regression) that a learned model must beat.

**Cloud masking**: removing pixels flagged as cloud, shadow, cirrus or snow using the QA_PIXEL band.

**Composite**: a single image built from many acquisitions, here the monthly per-pixel median.

**Counterfactual scenario**: model inputs deliberately changed (for example NDVI + 10%) to see how the output responds. It measures model sensitivity, not real-world causation.

**ERA5-Land**: ECMWF reanalysis of land surface variables at about 9 km resolution.

**LST (land surface temperature)**: radiometric temperature of the land surface, derived from the thermal band. Distinct from air temperature.

**Local Climate Zones (LCZ)**: a classification of urban form and cover used to describe thermal environments (Stewart and Oke, 2012).

**LSTM**: a recurrent neural network with gated memory, used here to forecast from a sequence of monthly inputs.

**N_OBS**: number of clear observations that contributed to a pixel in a monthly composite.

**NDVI**: normalized difference vegetation index, (NIR - Red) / (NIR + Red).

**Persistence forecast**: predict that next month equals this month.

**Seasonal climatology**: predict next month as the average of that calendar month in the training years.

**Surface urban heat island (SUHI)**: the LST difference between urban and surrounding rural or vegetated surfaces.

**Temporal split**: train on earlier years and test on later ones, so no future information leaks into training.

**ST_QA**: Landsat Level-2 per-pixel surface temperature uncertainty, in units of 0.01 K.

**Anomaly**: a value minus its cell-and-calendar-month climatology from the training years.

**Cross-sectional slope**: slope of LST on NDVI across cells within one month. It mixes the effect of vegetation with everything correlated with it (built form, imperviousness, moisture).

**Local response**: the model's change in LST for a change in a cell's own NDVI while the city-wide monthly mean is held fixed. The quantity relevant to greening scenarios.

**Hierarchical (two-level) model**: LST = climatology + month-level anomaly (from ERA5, linear, about 55 training months) + within-month spatial anomaly (from NDVI and cell context, many pixels).

**Month-cluster bootstrap**: resampling whole months with replacement, because pixels within a month share weather and overpass conditions and are not independent.

**Held-out spatial blocks**: 4.8 km blocks excluded from model fitting to test generalisation to unseen locations.

**JRC Global Surface Water occurrence**: percent of time a pixel was water, used to mask rivers, ponds and wetlands.

**Attenuation bias**: shrinkage of a regression slope towards zero caused by noise in the predictor (here NDVI).