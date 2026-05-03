# Covertype: MLP Vanilla vs MLP+EAAR (classification, 100k subset, 3 seeds)

Source:
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar3.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 3 |
| ΔAccuracy mean (EAAR - vanilla) | +0.000650 |
| ΔAccuracy std | 0.001784 |
| ΔMacro-F1 mean | -0.004929 |
| ΔMacro-F1 std | 0.003729 |
| ΔCE mean | +0.000542 |
| ΔCE std | 0.003465 |
| Vanilla AUC gap (CE) mean | 2.3461 |
| EAAR AUC gap (CE) mean | 2.4187 |
| ΔAUC gap (CE) mean | +0.0726 |
| Faithfulness wins/losses (AUC gap CE) | 3 / 0 |
| Vanilla top/random (CE) mean | 12.8095 |
| EAAR top/random (CE) mean | 13.1297 |

Interpretation:
- EAAR improves internal faithfulness on all 3 seeds (`AUC gap CE` and `top/random` both higher).
- Predictive quality is essentially preserved on this quick stage (`ΔAccuracy` near zero).
- This supports extension of EAAR beyond regression to multiclass tabular classification.

## Per-seed deltas

| Seed | ΔAccuracy | ΔMacro-F1 | ΔCE | Vanilla AUC gap CE | EAAR AUC gap CE | ΔAUC gap CE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.002600 | -0.003890 | -0.000422 | 2.3701 | 2.5028 | +0.1328 |
| 43 | -0.000900 | -0.001829 | +0.004387 | 2.3514 | 2.3821 | +0.0307 |
| 44 | +0.000250 | -0.009067 | -0.002339 | 2.3168 | 2.3712 | +0.0544 |
