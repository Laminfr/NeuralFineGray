# Survival Stacking

This folder contains experiments for survival stacking models that combine multiple base learners (CoxPH, DeepSurv, RSF, XGBoost, NFG) with optional tabular foundation model embeddings.

## Overview

Evaluates stacking ensemble methods for survival analysis using 5-fold cross-validation on METABRIC, PBC, and SUPPORT datasets. Models are tested with raw features and deep embeddings (TabICL/TabPFN/TARTE).

## Step-by-Step Guide

### 1. Run Full Benchmark

```bash
python survivalStacking/run_full_benchmark.py --dataset METABRIC
python survivalStacking/run_full_benchmark.py --dataset PBC
python survivalStacking/run_full_benchmark.py --dataset SUPPORT
```

### 2. Check Results

Results are saved as JSON files in:
```
results/survival_stacking/
```

Example files:
- `METABRIC_full_benchmark_5fold.json`
- `PBC_full_benchmark_5fold.json`
- `SUPPORT_full_benchmark_5fold.json`

Model files are saved in:
```
results/survival_stacking/models/<DATASET>/
```

### 3. Visualize Results

Generate plots:

```bash
python survivalStacking/visualize_benchmark_results.py --dataset METABRIC
python survivalStacking/visualize_benchmark_results.py --dataset PBC
python survivalStacking/visualize_benchmark_results.py --dataset SUPPORT
```

Plots are saved to:
```
results/survival_stacking/plots/
```

### 4. Compute Statistical Significance

Compare models statistically:

```bash
python survivalStacking/compute_significance.py --dataset METABRIC
```

## Logs

SLURM logs are saved in:
```
survivalStacking/logs/
```
