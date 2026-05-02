## SML2010: EA-minimal vs Vanilla Post-hoc (permute deletion, 10 seeds)

Source files:
- `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt_permute_final_v2.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt_permute_vanilla_v2.json`

| Metric | Vanilla post-hoc | EA-minimal |
|---|---:|---:|
| AUC deletion top | 0.1258 | 0.4496 |
| AUC deletion random | 0.2280 | 0.2280 |
| AUC deletion bottom | 0.4808 | 0.0268 |
| AUC(top-bottom) gap | -0.3550 | 0.4228 |
| AUC top/random ratio | 0.6353 | 2.2985 |
| AUC top/bottom ratio | 0.2758 | 28.7220 |
| Relative AUC deletion top | 0.0003116 | 0.0011151 |
| Relative AUC deletion bottom | 0.0011940 | 0.0000663 |

Interpretation:
- Vanilla ranking is functionally inconsistent (`top < random < bottom`).
- EA-minimal ranking is functionally consistent (`top > random > bottom`).
- Main article thesis should focus on faithfulness/compactness with preserved quality, not on large R² gains.

---

## Non-fast confirmation (SML2010, 5 seeds: 42–46, permute, random_trials=20)

Source files:
- `results/multiseed_config_sml2010_ea_minimal_sml_ea5_full.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea5_full_permute_vanilla_v3.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea5_full_permute_shap_v3.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea5_full_permute_final_v3.json`

SHAP apply status on this non-fast run:
- `shap`: 2/5
- `vanilla_fallback`: 3/5

Faithfulness by evaluation mode:

| Eval mode | AUC top | AUC random | AUC bottom | AUC gap (top-bottom) | AUC top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla-only (`eval=vanilla`) | 0.0796 | 0.2143 | 0.4947 | -0.4151 | 0.3687 |
| EA-only (`eval=shap`) | 0.4788 | 0.2143 | 0.0122 | +0.4666 | 2.1792 |
| Policy final (`eval=final`) | 0.2126 | 0.2143 | 0.3063 | -0.0937 | 0.9886 |

95% CI for `AUC gap (top-bottom)`:
- Vanilla-only: `[-0.5574, -0.2727]`
- EA-only: `[+0.3234, +0.6099]`
- Policy final (mixed with fallback): `[-0.5141, +0.3267]`

Takeaway:
- In non-fast mode, **EA-only** keeps the same functional behavior (`top > random > bottom`) and clearly dominates vanilla on faithfulness.
- Mixed policy output (`final`) is weaker due to fallback blending; article tables should report both `final` policy behavior and `shap-only` faithfulness explicitly.
