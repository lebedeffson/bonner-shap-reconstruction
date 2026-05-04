# EAAR Q1/Q2 Final Pack (2026-05-04)

## Added in this cycle

1. `neg_core_v2_internal_only_no_fallback` (mechanism sanity without policy collapse)
2. `ROAR-lite` retrain block on SML2010 (surrogate)
3. `SAGE` baseline on SML2010 (ANFIS checkpoints)

---

## 1) Internal-only no-fallback

Files:
- `results/ablation_neg_core_v2_internal_only_no_fallback_nf3_q1q2.md`
- `results/ablation_neg_core_v2_internal_only_no_fallback_s1_q1q2.md`

Key:
- `fallback_rate=0.0` across variants.
- `s1` deletion snapshot:
  - `full_rho1`: `AUC gap=0.3858`
  - `random_target`: `0.3240`
  - `sparsity_only`: `0.3011`

Takeaway:
- policy-collapse artifact removed;
- unique `q_err` contribution still not fully isolated vs sparsity.

---

## 2) ROAR-lite (surrogate retrain, 3 seed, k=1,2)

Files:
- `results/roar_lite_sml2010_surrogate_3seed_20260504.md`
- `results/roar_lite_sml2010_surrogate_3seed_20260504.csv`

Mean `roar_gap_top_minus_bottom`:
- `eaar_internal`: `k1=-0.00019`, `k2=-0.00039`
- `permutation`: `k1=+0.00091`, `k2=+0.00231`
- `vanilla_gradient`: `k1=-0.00247`, `k2=+0.00120` (high variance)

Takeaway:
- retrain signal is weak/noisy on small N-seed;
- permutation is still the most stable in ROAR-lite.

---

## 3) SAGE baseline (SML2010, 3 seed)

Files:
- `results/sage_sml2010_3seed_20260504.md`
- `results/sage_sml2010_3seed_20260504_meta.csv`
- `results/sage_sml2010_3seed_20260504_features.csv`

Key:
- `corr(SAGE, EAAR_internal)` per-seed: `+0.284`, `+0.274`, `+0.237`
- top mean SAGE features:
  - `meteo_exterior_sol_sud`
  - `lighting_comedor_sensor`
  - `meteo_exterior_sol_oest`

Takeaway:
- SAGE baseline is now explicitly present;
- alignment with EAAR-internal is consistently positive but moderate.

---

## Submission-safe boundary

- Stronger claim: EAAR repairs internal faithfulness in ANFIS/MLP setting.
- Bounded claim: mechanism-specific separation (`q_err` vs compactness) remains partial.
- For strict Q1/Q2: expand ROAR/KAR retrain seeds and deepen mechanism ablation.
