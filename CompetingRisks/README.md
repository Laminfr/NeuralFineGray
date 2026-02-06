# Competing Risks Analysis

This folder contains benchmarking experiments for competing risks models including discrete-time multiclass approaches, hybrid models, NFG wrappers, and stacking methods. Evaluates multiple competing risks approaches across benchmark datasets using 5-fold cross-validation to compare discrimination and calibration metrics.

## Methods

1. **Stacking**: Multi-class XGBoost on discrete-time person-period data
2. **NFG**: Neural Fine-Gray (deep learning)  
3. **Hybrid**: NFG embeddings + XGBoost

## Datasets

- `SYNTHETIC_COMPETING`: Auto-downloaded synthetic data (2 competing events)
- `SEER_competing_risk`: Real cancer data (requires `datasets/seer/seernfg.csv`)

## Step-by-Step Guide

### 1. Run Benchmark

```bash
# Run on synthetic data
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING

# Run with SEER (if you have the data file)
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING SEER_competing_risk
```

### 2. Check Results

Results are saved as JSON files in:
```
results/competing_risks/
```

Example files:
- `full_benchmark_5fold.json`
- `seer_benchmark_5fold.json`
- `synthetic_competing_benchmark_5fold.json`

### 3. Visualize Results

Generate plots:

```bash
python CompetingRisks/visualize_results.py
```

Plots are saved to:
```
results/competing_risks/plots/
```

## Logs

SLURM logs are saved in:
```
CompetingRisks/logs/
```

