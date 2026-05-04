# EAAR TISU Final Results Pack (2026-05-04)

Короткий финальный пакет для статьи и защиты результатов без оверклейма.

## 1) Main ANFIS result (SML2010, non-fast, stability guard)
- `results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md`
- Core numbers:
  - Vanilla gradient `AUC gap = -0.4151`
  - Vanilla permutation `AUC gap = +0.5661`
  - EAAR internal `AUC gap = +0.4666`
  - old final policy `AUC gap = -0.1739`
- Stable-only predictive shift: `ΔR² = -0.000087`

## 2) Time-block robustness
- `results/sml2010_faithfulness_mode_compare_timeblock5_nf_20260503.md`
- `results/sml2010_faithfulness_mode_compare_timeblock3_nf_20260503.md`
- Main pattern is stable on temporal split:
  - vanilla gradient remains negative
  - EAAR internal remains positive
  - permutation baseline remains stronger

## 3) MLP portability (regression, SML2010)
- 5-seed: `results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md`
- 10-seed: `results/mlp_eaar_vs_vanilla_sml2010_10seed_20260504.md`
- 10-seed summary:
  - `ΔAUC gap ≈ +1.90` (~+10.95%)
  - `ΔR² ≈ -0.000285`
  - positive direction preserved, strict p<0.05 not reached

## 4) Classification portability (Covertype)
- 100k/5 seed:
  - `results/covtype_mlp_eaar_vs_vanilla_5seed_20260503.md`
  - `results/significance_covtype_mlp_eaar_vs_vanilla_auc_gap_ce_5seed_pc.json`
- 300k/3 seed:
  - `results/covtype300k_mlp_eaar_vs_vanilla_3seed_20260503.md`
- Per-class delta:
  - `results/covtype_per_class_delta_5seed_20260504.md`
- Per-class full summary:
  - `results/covtype_per_class_full_5seed_20260504.md`
- Conclusion: initial classification evidence, macro-F1 drift is small and explicitly monitored.

## 5) Extra model check: ResMLP (Energy, 5 seed)
- `results/resmlp_eaar_vs_vanilla_energy_5seed_20260504.md`
- `results/significance_energy_resmlp_eaar_vs_vanilla_auc_gap_5seed.json`
- Boundary-case: mixed transfer, no stable mean gain on this pair.

## 6) Negative controls status
- Baseline non-fast controls:
  - `results/ablation_neg3_nonfast_controls_20260503.md`
- Expanded snapshot:
  - `results/ablation_neg_controls_expanded_20260503.md`
- Latest fast completion runs:
  - `results/ablation_neg_core_v2_fast3_final.md`
  - `results/ablation_gamma_sweep_v2_fast3_final.md`
  - `results/ablation_neg_core_v2_fast3_final_alignment_20260504.md`
- Internal-only mechanism run (order-fix, non-fast, 3 seed):
  - `results/ablation/ablation_manifest_config_sml2010_ea_minimal_qerr_focus_orderfix_nf3.json`
  - `results/qerr_focus_orderfix_nf3_summary.md`
  - key: `fallback_rate=0.0` for all variants; `full_rho1` is not the top AUC-gap variant (`full=0.5144`, `random=0.4566`, `sparsity=0.5288`, `task=0.4744`)
- Honest reading:
  - random target is weaker than full in part of runs;
  - `sparsity_only` remains competitive;
  - unique `q_err` contribution is **partially** supported and remains a limitation.
  - fast `neg_core_v2/gamma_sweep` behaved almost identically across variants, so they are treated as stability/smoke evidence, not decisive mechanism evidence.

## 7) Timing and reproducibility
- Timing:
  - `results/timing_table_q1_20260503.md`
- Protocol:
  - `results/experimental_protocol_sml2010_eaar_q1.md`
- Baselines:
  - `results/predictive_baselines_full_metrics_20260503.md`
- Manifest:
  - `results/results_manifest.json`

## 8) Safe claim boundary (for manuscript)
1. EAAR repairs internal attribution sign/pattern vs vanilla gradient.
2. EAAR does not claim SOTA predictive accuracy.
3. EAAR does not claim superiority over permutation post-hoc baseline.
4. Classification and additional model results are initial/conditional portability evidence.

## 9) Submission-ready wording
- Use: `initial evidence`, `preliminary classification evidence`, `major-revision package`.
- Avoid: `proved general applicability`, `production-ready final policy`, `better than permutation`.

## 10) Reviewer-ready facts
- `results/reviewer_response_facts_20260504.md`
