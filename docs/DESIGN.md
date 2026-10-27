# Design: Urban Heat Island Forecasting (revised after Stage 3-4 results)

Author: Tamojeet Pal, @iTamojeet

## Research question

How does neighbourhood-scale land surface temperature (LST) in Kolkata respond to vegetation (NDVI), and how much of the cross-sectional NDVI-LST relationship survives as a local, within-cell response usable for greening scenarios?

## Hypotheses (revised)

- H1 (original, not supported): an LSTM using past LST, NDVI and meteorology beats persistence, climatology and linear baselines for next-month LST. Stage 3-4 showed past anomalies carry no skill beyond climatology and flexible models overfit month-level weather (about 55 independent training months).
- H1' (reframed): a nonlinear model of the within-month spatial pattern, driven by NDVI and cell context, beats climatology and linear models on the spatial component of the error.
- H2: the model's local NDVI response has the same sign as the empirical cross-sectional slope but a smaller magnitude, because the cross-sectional slope confounds NDVI with urban form. Scenario outputs are model sensitivity, not causal estimates.

## Data

Landsat 8/9 Collection 2 Level-2 monthly median LST and NDVI, 2014-2025, Kolkata core, 30 m aggregated to 120 m; ERA5-Land monthly area-mean skin temperature, 2 m temperature, precipitation; JRC Global Surface Water occurrence for the water mask. Usable months: about 88 of 144, with no usable July-September observations.

## Stages (folder numbers)

1. `01_data_acquisition`: monthly composites, ERA5, coverage log, water mask.
2. `02_preprocessing`: usable-month filter, cold-outlier masking, 120 m cube, land mask, spatial blocks, per-image NDVI-LST slopes.
3. `03_baselines`: persistence, climatology, linear, linear + concurrent ERA5; temporal split (train <= 2022, val 2023, test 2024-25) and held-out spatial blocks.
4. `04_models`: gradient boosting (prev / concurrent inputs) and the two-level hierarchical model with an exact month/spatial error decomposition.
5. `05_counterfactual`: local NDVI scenarios (+10/20/30%) and month-cluster bootstrap intervals for local vs cross-sectional response.
6. Write-up: README, figures, limitations.

## Known limitations

No July-September coverage; monthly composites rest on 1-6 clear overpasses; month-level error dominates and is weakly predictable; NDVI measurement noise attenuates the local response while confounding inflates the cross-sectional one, so the truth lies between; no ground-truth for scenarios.