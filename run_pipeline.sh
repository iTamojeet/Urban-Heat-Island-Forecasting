#!/usr/bin/env bash
# Runs stages 02-06 in order and saves the combined output to data/processed/run_log.txt.
# Requires stage 01 outputs in data/raw (see README).
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p data/processed docs/figures
{
  python 02_preprocessing/build_cube.py
  python 03_baselines/baselines.py
  python 04_models/gbm.py
  python 04_models/hierarchical.py
  python 05_counterfactual/ndvi_scenarios.py
  python 05_counterfactual/bootstrap_ci.py
  python 05_counterfactual/make_final_figure.py
  python 06_cool_island/cool_island.py
} 2>&1 | tee data/processed/run_log.txt
