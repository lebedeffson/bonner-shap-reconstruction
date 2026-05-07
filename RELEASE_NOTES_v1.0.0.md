# Release Notes — v1.0.0

## Summary
v1.0.0 is the first formal release of the SHAP-regularized Bonner spectrum reconstruction pipeline.
It focuses on practical reliability: measurable regularization contribution, faithfulness checks, and reproducible reporting.

## Highlights
- Full SHAP training path (exact estimator mode).
- Practical PASS/FAIL gate for release readiness.
- Internal importance export aligned with training-time regularization mechanics.
- Faithfulness and alignment reports integrated into the standard workflow.
- Official README with method equations and release artifact policy.

## Core Practical Outcome
- Regularization is no longer treated as negligible/noise in strong-check runs.
- Readiness evaluation is now explicit and scriptable.

## Key Commands
Train:
```bash
python train.py --config configs/config_shap_exact_alignment_push_quickcheck.yaml --tag release_run
```

Faithfulness:
```bash
PYTHONPATH=. python scripts/report_faithfulness_top_random_bottom.py \
  --summary results/training_summary_<timestamp>.json \
  --k-max 4 --random-trials 20 --masking permute
```

Alignment:
```bash
PYTHONPATH=. python scripts/report_importance_alignment.py \
  --ref results/feature_importance_shap_<timestamp>.csv \
  --cand results/feature_importance_internal_<timestamp>.csv \
  --label-ref shap_target --label-cand internal_grad --k 3,5
```

Practical Gate:
```bash
PYTHONPATH=. python scripts/practical_readiness_gate.py \
  --summary results/training_summary_<timestamp>.json \
  --faithfulness results/faithfulness_<timestamp>.json \
  --alignment results/importance_alignment_<timestamp>.json
```

## Artifact Policy
Commit only lightweight outputs (`.md`, `.json`, metrics `.csv`).
Do not commit heavy artifacts (`.pt`, `.npy`, `.png`, `.pdf`).

