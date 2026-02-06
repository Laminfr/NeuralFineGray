# NeuralFineGray

Survival analysis framework with Neural Fine-Gray models, Survival Stacking, and foundation model embeddings (TabICL, TabPFN, TARTE).

## Requirements

```bash
# Create and activate environment
conda create -n nfg python=3.10
conda activate nfg

# Install core dependencies
pip install torch numpy pandas scikit-learn lifelines pycox xgboost

# For TabICL embeddings
pip install tabicl

# For TabPFN embeddings  
pip install tabpfn

# For TARTE embeddings
pip install tarte-ai
```

Set HuggingFace token for TabPFN access:
```bash
export HF_TOKEN="your_token_here"
# Or create .env file with: HF_TOKEN=your_token_here
```

## Available Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| METABRIC | Binary survival | Breast cancer, ~2000 samples |
| SUPPORT | Binary survival | ICU mortality, ~9000 samples |
| PBC | Binary survival | Primary biliary cirrhosis, ~418 samples |
| GBSG | Binary survival | German breast cancer study |
| SYNTHETIC_COMPETING | Competing risks | Synthetic data with 2 event types |
| SEER_competing_risk | Competing risks | Cancer registry (requires local file) |

---

## 1. Baseline Experiments (Raw Features)

Run individual survival models on raw features with hyperparameter search.

### Command Line

```bash
# Single model on single dataset
python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw
python -m experiments.run_experiment --dataset SUPPORT --model deepsurv --mode raw
python -m experiments.run_experiment --dataset PBC --model rsf --mode raw
python -m experiments.run_experiment --dataset METABRIC --model xgboost --mode raw
python -m experiments.run_experiment --dataset SUPPORT --model nfg --mode raw

# Run on all datasets
python -m experiments.run_experiment --dataset all --model coxph --mode raw
```

### Models

| Model | Flag | Description |
|-------|------|-------------|
| CoxPH | --model coxph | Cox Proportional Hazards |
| DeepSurv | --model deepsurv | Neural network survival |
| RSF | --model rsf | Random Survival Forest |
| XGBoost | --model xgboost | Gradient boosting survival |
| NFG | --model nfg | Neural Fine-Gray |

### Options

```
--dataset     Dataset name (METABRIC, SUPPORT, PBC, GBSG, all)
--model       Model type (coxph, deepsurv, rsf, xgboost, nfg)
--mode        Feature mode (raw, tabpfn)
--fold        Specific CV fold to run (0-4), omit for all folds
--grid-search Number of hyperparameter search iterations (default: 100)
--seed        Random seed (default: 0)
--output-dir  Custom output directory
```

### SBATCH

```bash
sbatch examples/baselines/run_all_baselines.sbatch
```

---

## 2. Experiments with TabPFN Embeddings

Run same models but with TabPFN-extracted feature embeddings.

```bash
python -m experiments.run_experiment --dataset METABRIC --model coxph --mode tabpfn
python -m experiments.run_experiment --dataset SUPPORT --model deepsurv --mode tabpfn
python -m experiments.run_experiment --dataset PBC --model rsf --mode tabpfn
python -m experiments.run_experiment --dataset METABRIC --model xgboost --mode tabpfn
python -m experiments.run_experiment --dataset SUPPORT --model nfg --mode tabpfn
```

---

## 3. Survival Stacking Benchmark

Comprehensive benchmark comparing stacking methods with different embeddings.

### Methods Compared

1. SurvStack-Raw: Stacking with XGBoost on raw features
2. SurvStack-TabICL-Emb: Stacking with XGBoost on TabICL embeddings + raw
3. SurvStack-TabPFN-Emb: Stacking with XGBoost on TabPFN embeddings + raw
4. SurvStack-TabICL: Stacking with TabICL classifier
5. SurvStack-TabPFN: Stacking with TabPFN classifier
6. Baselines: CoxPH, XGBoost, DeepSurv (non-stacked)

### Command Line

```bash
# Single dataset
python -m survivalStacking.run_full_benchmark --dataset METABRIC --cv 5

# All datasets
python -m survivalStacking.run_full_benchmark --dataset all --cv 5

# Custom intervals
python -m survivalStacking.run_full_benchmark --dataset PBC --n_intervals 15
```

### Options

```
--dataset      Dataset (METABRIC, SUPPORT, PBC, all)
--cv           Number of CV folds (default: 5)
--n_intervals  Time intervals for stacking (default: 20)
--weighting    Weighting strategy (adaptive, uniform, none)
--verbose      Print progress
```

### SBATCH

```bash
# Submit job
sbatch survivalStacking/run_full_benchmark.sbatch METABRIC 5

# Or for all datasets
sbatch survivalStacking/run_full_benchmark.sbatch all 5
```

Results saved to: results/survival_stacking/

Visualizations auto-generated via:
```bash
python -m survivalStacking.visualize_benchmark_results
```

---

## 4. Competing Risks Benchmark

Benchmark for competing risks data with multiple event types.

### Methods

1. Stacking: Multi-class survival stacking with XGBoost
2. NFG: Pure Neural Fine-Gray for competing risks
3. Hybrid: NFG embeddings + XGBoost stacking

### Command Line

```bash
# Synthetic data
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING --n-folds 5

# SEER (requires local file at datasets/seer/seernfg.csv)
python -m CompetingRisks.run_benchmark --datasets SEER_competing_risk --n-folds 5

# All phases on all datasets
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING SEER_competing_risk --phases stacking nfg hybrid
```

### Options

```
--datasets   Datasets to evaluate (SYNTHETIC_COMPETING, SEER_competing_risk)
--n-folds    Number of CV folds (default: 5)
--phases     Methods to run (stacking, nfg, hybrid)
--output-dir Output directory
--quiet      Suppress output
```

### SBATCH

```bash
sbatch CompetingRisks/run_benchmark.sbatch
```

Results saved to: results/competing_risks/

Visualizations via:
```bash
python -m CompetingRisks.visualize_results
```

---

## 5. Embedding Extraction (Standalone)

Extract embeddings from foundation models for use in other pipelines.

### TabICL Embeddings

```python
from datasets.tabicl_embeddings import apply_tabicl_embedding

X_train_emb, X_val_emb, X_test_emb, feature_names = apply_tabicl_embedding(
    X_train, X_val, X_test, 
    E_train,  # event labels
    feature_names=original_feature_names,
    use_deep_embeddings=True,
    concat_with_raw=True,  # raw + embeddings
    verbose=True
)
```

### TabPFN Embeddings

```python
from survivalStacking.tools import apply_tabpfn_embeddings

X_train_emb, X_val_emb, X_test_emb = apply_tabpfn_embeddings(
    X_train, X_val, X_test,
    E_train,
    concat_with_raw=True
)
```

### TARTE Embeddings

```python
from datasets.tarte_embeddings import apply_tarte_embedding

X_train_emb, X_val_emb, X_test_emb, feature_names = apply_tarte_embedding(
    X_train, X_val, X_test,
    E_train,
    feature_names=original_feature_names,
    use_deep_embeddings=True,
    concat_with_raw=True
)
```

---

## Project Structure

```
NeuralFineGray/
├── experiments/           # Baseline experiments
│   ├── run_experiment.py  # Unified experiment runner
│   └── experiment.py      # Experiment classes
├── survivalStacking/      # Survival stacking methods
│   ├── run_full_benchmark.py
│   ├── stacking_model.py
│   └── discrete_time.py
├── CompetingRisks/        # Competing risks models
│   ├── run_benchmark.py
│   └── discrete_time_multiclass.py
├── datasets/              # Data loading and embeddings
│   ├── datasets.py
│   ├── tabicl_embeddings.py
│   └── tarte_embeddings.py
├── nfg/                   # Neural Fine-Gray implementation
├── coxph/                 # Cox PH implementation
├── deepsurv/              # DeepSurv implementation
├── rsf/                   # Random Survival Forest
├── xgb_survival/          # XGBoost survival
├── metrics/               # Evaluation metrics
├── core/                  # Shared utilities
└── results/               # Output directory
```

---

## Quick Start Examples

### Run all baselines on METABRIC

```bash
for model in coxph deepsurv rsf xgboost nfg; do
    python -m experiments.run_experiment --dataset METABRIC --model $model --mode raw
done
```

### Full stacking benchmark on PBC

```bash
python -m survivalStacking.run_full_benchmark --dataset PBC --cv 5 --verbose
```

### Competing risks on synthetic data

```bash
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING --phases stacking nfg hybrid
```

---

## Output

All experiments save results as JSON files with:
- Per-fold metrics (C-index at quantiles, IBS)
- Mean and std across folds
- Training times
- Model configurations

Visualization scripts generate:
- Bar charts comparing methods
- Box plots of CV distributions
- Tables in LaTeX format
