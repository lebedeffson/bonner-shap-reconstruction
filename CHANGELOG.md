# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-05-07

### Added
- Practical readiness gate: `scripts/practical_readiness_gate.py`
- Heavy artifact cleanup utility: `scripts/cleanup_results_heavy.py`
- Internal importance export aligned with training-time regularization:
  - `feature_importance_internal_*.csv`
- New release-ready configs:
  - `configs/config_shap_exact_alignment_push_quickcheck.yaml`
  - `configs/config_shap_exact_strong_fixed.yaml`
  - `configs/config_shap_exact_strong_fixed_quickcheck.yaml`
  - `configs/config_shap_exact_strong_signal.yaml`

### Changed
- SHAP regularization signal strengthened and normalized by final contribution ratio.
- Adaptive component weighting revised to avoid collapse of meaningful components.
- `report_faithfulness_top_random_bottom.py` now supports explicit `--importance-path`.
- README rewritten in official release format with mathematical method section and artifact policy.

### Fixed
- Practical alignment/evaluation mismatch resolved by using internal importance consistent with training loss.
- Reduced risk of weak regularization behaving as near-noise in strong-check configurations.

### Notes
- This release targets the SHAP-regularized Bonner/spectra pipeline.
- EAAR line is maintained in a separate repository.

