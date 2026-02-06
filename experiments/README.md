# Experiments

Unified experiment runner for survival analysis models with hyperparameter search.

## Quick Start

```bash
cd /vol/miltank/users/sajb/Project/NeuralFineGray/experiments

# Run CoxPH on all datasets
sbatch run_experiment.sbatch all coxph raw

# Run specific model on specific dataset
sbatch run_experiment.sbatch METABRIC deepsurv raw
sbatch run_experiment.sbatch SUPPORT rsf raw
sbatch run_experiment.sbatch PBC xgboost raw
sbatch run_experiment.sbatch METABRIC nfg raw

# Run with TabPFN embeddings
sbatch run_experiment.sbatch METABRIC coxph tabpfn
```

## Usage

```bash
sbatch run_experiment.sbatch DATASET MODEL MODE [GRID] [SEED] [FOLD]
```

### Arguments (positional)

| Argument | Values | Description |
|----------|--------|-------------|
| DATASET | METABRIC, SUPPORT, PBC, all | Dataset to run on |
| MODEL | coxph, deepsurv, rsf, xgboost, nfg | Survival model |
| MODE | raw, tabpfn | Feature mode (raw or TabPFN embeddings) |
| GRID | integer (default: 100) | Random search iterations |
| SEED | integer (default: 0) | Random seed |
| FOLD | integer or empty | Specific fold (0-4), or empty for all |

## Models

| Model | Description |
|-------|-------------|
| coxph | Cox Proportional Hazards (lifelines) |
| deepsurv | Deep Cox neural network (PyTorch) |
| rsf | Random Survival Forest (scikit-survival) |
| xgboost | XGBoost with Cox objective |
| nfg | Neural Fine-Gray |

## Examples

```bash
# All models on METABRIC with raw features
for model in coxph deepsurv rsf xgboost nfg; do
    sbatch run_experiment.sbatch METABRIC $model raw
done

# All models on all datasets
for model in coxph deepsurv rsf xgboost nfg; do
    sbatch run_experiment.sbatch all $model raw
done

# Single fold for quick testing
sbatch run_experiment.sbatch METABRIC coxph raw 50 0 0
```

## Output

Results saved to `results/` as CSV files:
- `{DATASET}_raw_{MODEL}.csv` - raw features
- `{DATASET}_tabpfn_{MODEL}.csv` - TabPFN embeddings

Logs saved to `logs/`:
- `exp_exp_run-{JOBID}.out` - stdout
- `exp_exp_run-{JOBID}.err` - stderr

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| C-index | Concordance index (discrimination, higher is better) |
| IBS | Integrated Brier Score (calibration, lower is better) |

## Directory Structure

```
experiments/
├── run_experiment.py      # Unified experiment runner
├── run_experiment.sbatch  # SLURM submission script
├── experiment.py          # Experiment class definitions
├── results/               # Output CSV files
├── logs/                  # SLURM job logs
└── analysis/              # Jupyter notebooks for visualization
```

## Direct Python Usage

```bash
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# Run directly without SLURM
python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw
python -m experiments.run_experiment --dataset all --model xgboost --mode tabpfn --grid-search 50
```

### Python Arguments

```
--dataset, -d     Dataset name (required)
--model, -m       Model type (required)
--mode            Feature mode: raw or tabpfn (default: raw)
--fold, -f        Specific fold to run
--grid-search, -g Random search iterations (default: 100)
--seed, -s        Random seed (default: 0)
--output-dir, -o  Custom output directory
```
