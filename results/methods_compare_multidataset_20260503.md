# Сравнение методов на разных датасетах (R², 10 seed)

| Dataset | Method | R² mean | R² std |
|---|---|---:|---:|
| energy_efficiency | hgb | 0.994233 | 0.000799 |
| energy_efficiency | anfis_final_policy | 0.981759 | 0.004157 |
| energy_efficiency | anfis_vanilla | 0.981726 | 0.004131 |
| energy_efficiency | anfis_ea_raw | 0.981424 | 0.004421 |
| energy_efficiency | et | 0.979315 | 0.003042 |
| energy_efficiency | rf | 0.979133 | 0.002907 |
| energy_efficiency | mlp | 0.974179 | 0.033878 |
| naval_propulsion | et | 0.996744 | 0.000313 |
| naval_propulsion | hgb | 0.993697 | 0.000381 |
| naval_propulsion | rf | 0.992651 | 0.001696 |
| naval_propulsion | mlp | 0.978573 | 0.007681 |
| naval_propulsion | anfis_final_policy | 0.858550 | 0.061085 |
| naval_propulsion | anfis_vanilla | 0.858500 | 0.061081 |
| naval_propulsion | anfis_ea_raw | 0.858129 | 0.061112 |
| sml2010 | et | 0.995714 | 0.000485 |
| sml2010 | hgb | 0.994505 | 0.000436 |
| sml2010 | mlp | 0.988333 | 0.001343 |
| sml2010 | rf | 0.988072 | 0.000889 |
| sml2010 | anfis_ea_raw | 0.952482 | 0.008253 |
| sml2010 | anfis_final_policy | 0.952482 | 0.008253 |
| sml2010 | anfis_vanilla | 0.952475 | 0.008257 |

Примечание: для ANFIS использованы сохраненные multiseed-файлы EA-minimal; для классических baseline запущен `scripts/run_tabular_baselines.py`.