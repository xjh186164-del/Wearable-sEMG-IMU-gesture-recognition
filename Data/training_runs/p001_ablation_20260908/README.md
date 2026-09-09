# Controlled modality and fusion ablation

This directory contains the additional experiment used to determine how much
performance came from sEMG, IMU and multimodal fusion. It also tests whether
trainable class-dependent fusion improves upon fixed 50:50 fusion and whether
physics-informed initial weights are necessary.

## Experiment design

All retrained variants use the published `p001_windows.npz` dataset, the same
session-level training/validation/test split, and the three fixed seeds
20260821, 20260822 and 20260823. The five variants are:

- `emg_only`: sEMG encoder and classifier only;
- `imu_only`: IMU encoder and classifier only;
- `fixed_50_50`: both branches with fixed equal logit fusion;
- `learned_neutral`: trainable class weights initialised to 0.5;
- `learned_physics`: trainable class weights initialised using the expected
  physical contribution of the two modalities.

## Aggregate held-out results

| Variant | Macro-F1, mean +/- SD | Mean session Macro-F1 | Target-trial accuracy |
|---|---:|---:|---:|
| sEMG only | 85.33 +/- 2.89% | 85.57% | 91.67% |
| IMU only | 94.09 +/- 0.96% | 94.03% | 98.21% |
| Fixed 50:50 fusion | 99.64 +/- 0.34% | 99.63% | 100.00% |
| Learned fusion, neutral start | 99.65 +/- 0.32% | 99.65% | 100.00% |
| Learned fusion, physics-informed start | 99.67 +/- 0.03% | 99.66% | 100.00% |

Multimodal fusion substantially outperformed either isolated branch. The three
fused variants performed similarly, so these data do not establish a material
accuracy advantage for learned fusion over fixed equal fusion. The lower
cross-seed variability of the physics-informed runs is limited evidence from
three repetitions and should not be generalised beyond this experiment.

## Files

- `aggregate_summary.csv`: mean and sample standard deviation across seeds;
- `run_summary.csv`: one concise row for each of the 15 retraining runs;
- `ablation_results.json`: complete post-hoc and retraining results;
- `posthoc_existing_checkpoint/`: diagnostic branch/fusion evaluation of the
  original selected checkpoint;
- `retrained/<variant>/seed_<seed>/best_model.pt`: selected checkpoint;
- `history.csv`: epoch-by-epoch training and validation history;
- `metrics.json`: validation and held-out window metrics;
- `grouped_test_metrics.json`: per-session and target-trial results;
- `test_predictions.csv`: held-out window predictions;
- `split.json`: exact session split and dataset linkage.

The 0.5 s test windows advance by 0.1 s and therefore overlap by 80%.
Window-level predictions are not independent trials; use the grouped session
and target-trial metrics when interpreting robustness.

## Reproduction command

From the repository root after installing the Python package:

```powershell
.\.venv-gesture\Scripts\gesture-ablation.exe `
  --dataset .\Data\datasets\p001_windows.npz `
  --baseline-checkpoint .\Data\training_runs\p001_dual_cnn\best_model.pt `
  --output-dir .\Data\training_runs\p001_ablation_reproduction `
  --device cuda `
  --seeds 20260821,20260822,20260823
```

Use a new output directory so that the published results are not overwritten.
