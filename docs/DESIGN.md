# Design: Urban Heat Island Forecasting

Author: Tamojeet Pal, @iTamojeet

## Research question

Does a sequence model over monthly vegetation and meteorological inputs forecast next-month land surface temperature (LST) better than simple baselines, and is its learned sensitivity to NDVI physically plausible?

## Hypotheses

- H1: An LSTM using past LST, NDVI and meteorology reduces next-month LST error relative to persistence, seasonal climatology and a linear NDVI + meteorology regression, evaluated on held-out future years.
- H2: The LSTM's response to counterfactual NDVI inputs is consistent in sign and rough magnitude with the empirical NDVI-LST slope measured directly from the observations. Counterfactual outputs are model sensitivity, not causal cooling estimates.

## Study area and data

- Area: Kolkata metropolitan core, bbox in `01_data_acquisition/config.py`, UTM 45N, 30 m grid.
- Target and predictors: Landsat 8/9 Collection 2 Level-2 monthly median LST and NDVI, 2014-2025, with per-pixel clear-observation counts.
- Dense covariates: ERA5-Land monthly area-mean skin temperature, 2 m temperature, precipitation.
- Known limitation: monsoon cloud cover leaves June-September sparse. Stage 1 measures this; Stage 2 handles it with explicit missing-value masks rather than interpolating the target.

## Stages

1. Data acquisition: monthly composites, ERA5-Land table, coverage log and coverage figure.
2. Preprocessing and alignment: stack to a (time, band, y, x) array, aggregate to the analysis grid, build masks and the NDVI-LST empirical relationship, define spatial blocks.
3. Baselines: persistence, seasonal climatology, linear regression.
4. Training: LSTM on Colab T4, temporal split (train 2014-2022, validation 2023, test 2024-2025), checkpoints to Drive.
5. Evaluation: error against baselines by season, spatial error maps, held-out spatial blocks as a secondary test.
6. Counterfactual NDVI analysis: +10/+20/+30% NDVI scenarios, compared with the empirical slope from Stage 2.

Split years are provisional until the Stage 1 coverage figure is reviewed.
