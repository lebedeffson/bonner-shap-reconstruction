# Q1/Q2 Extended Validation (2026-05-04)

## 1) Non-fast gamma sweep (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_gamma_sweep_v2_nf3_q1q2.json`
- `results/ablation_gamma_sweep_v2_nf3_q1q2.md`
- `results/ablation_gamma_sweep_v2_nf3_q1q2.csv`

Key:
- `task_only`: `ΔR²=-0.0594`, `AUC gap=0.5392`
- `gamma_x03/x10/x30/x100_rho1`: `ΔR²=-0.0556`, `AUC gap=0.5380` (почти одинаково)
- `fallback_rate=1.0` для всех вариантов

Conclusion:
- В текущем контуре чувствительность по gamma остаётся слабой (метрики почти совпадают).

## 2) Non-fast divergence sweep (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_divergence_sweep_v1_nf3_q1q2.json`
- `results/ablation_divergence_sweep_v1_nf3_q1q2.md`
- `results/ablation_divergence_sweep_v1_nf3_q1q2.csv`

Variants:
- `div_cosine_mse`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `div_js`: `ΔR²=-0.0588`, `AUC gap=0.5392`
- `div_mse`: `ΔR²=-0.0362`, `AUC gap=0.5392`
- `div_js_mse`: `ΔR²=-0.0283`, `AUC gap=0.5392`

Conclusion:
- По AUC gap различия минимальные; divergence choice пока не даёт явного separation.

## 3) Deep negative ablation core (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_neg_core_v2_nf3_q1q2.json`
- `results/ablation_neg_core_v2_nf3_q1q2.md`
- `results/ablation_neg_core_v2_nf3_q1q2.csv`

Variants:
- `full_rho1`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `random_target`: `ΔR²=-0.0536`, `AUC gap=0.5392`
- `shuffled_q_err`: `ΔR²=-0.0403`, `AUC gap=0.5380`
- `uniform_target`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `anti_q_err`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `sparsity_only`: `ΔR²=-0.0501`, `AUC gap=0.5392`
- `task_only`: `ΔR²=-0.0594`, `AUC gap=0.5392`

Conclusion:
- В non-fast unmasked режиме на этой конфигурации уникальный вклад `q_err` всё ещё не изолирован.
- Для Q1/Q2 остаются обязательными: `ROAR/KAR-lite` и `SAGE` (либо мощный replacement-блок).

