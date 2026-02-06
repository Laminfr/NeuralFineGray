# Competing Risks Benchmark

Compares 3 approaches for survival analysis with multiple competing event types.

## Methods

1. **Stacking**: Multi-class XGBoost on discrete-time person-period data
2. **NFG**: Neural Fine-Gray (deep learning)  
3. **Hybrid**: NFG embeddings + XGBoost

## Datasets

- `SYNTHETIC_COMPETING`: Auto-downloaded synthetic data (2 competing events)
- `SEER_competing_risk`: Real cancer data (requires `datasets/seer/seernfg.csv`)

## Run Experiment

```bash
# Submit to SLURM (runs SYNTHETIC_COMPETING by default)
sbatch CompetingRisks/run_benchmark.sbatch

# Or run directly
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING

# With SEER (if you have the data file)
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING SEER_competing_risk
```

## Results

Saved to `results/competing_risks/`:
- `{dataset}_benchmark_5fold.json` - Metrics per dataset
- `plots/` - Comparison charts

