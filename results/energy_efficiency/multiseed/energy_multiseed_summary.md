# Energy Multiseed Summary

  model  n_runs  mse_mean  mse_std  rmse_mean  rmse_std  mae_mean  mae_std  r2_mean   r2_std  r2_outputs_mean  r2_outputs_std  importance_entropy_mean  importance_entropy_std  importance_gini_mean  importance_gini_std  importance_stability_corr
vanilla       5  1.459688 0.131939   1.206937  0.054689  0.768181 0.034294 0.984558 0.002097         0.983808        0.002393                 1.977509                 0.01828              0.244025             0.027707                   0.277256
   shap       5  3.515941 2.065637   1.804300  0.510334  1.285778 0.296505 0.962334 0.022951         0.961016        0.023643                 1.550966                 0.19123              0.522468             0.094529                   0.870831

# Detail

  model  seed                                             summary_file      mse     rmse      mae       r2  r2_mean  importance_entropy  importance_gini
   shap    41    training_summary_20260502_123132_energy_shap_s41.json 7.274957 2.697213 1.704142 0.921777 0.919899            1.491852         0.558775
   shap    42    training_summary_20260502_123404_energy_shap_s42.json 1.948179 1.395772 0.980776 0.980210 0.979567            1.828535         0.358272
   shap    43    training_summary_20260502_123633_energy_shap_s43.json 2.197000 1.482228 1.118469 0.977754 0.976879            1.240183         0.647942
   shap    44    training_summary_20260502_123853_energy_shap_s44.json 1.933894 1.390645 1.046225 0.980198 0.979959            1.570130         0.541753
   shap    45    training_summary_20260502_124133_energy_shap_s45.json 4.225675 2.055645 1.579279 0.951730 0.948777            1.624131         0.505599
vanilla    41 training_summary_20260502_122919_energy_vanilla_s41.json 1.499303 1.224460 0.781727 0.983879 0.983334            2.008314         0.200276
vanilla    42 training_summary_20260502_123148_energy_vanilla_s42.json 1.458184 1.207553 0.784227 0.985188 0.984444            1.973851         0.255238
vanilla    43 training_summary_20260502_123422_energy_vanilla_s43.json 1.257560 1.121410 0.729439 0.987266 0.986644            1.985170         0.224787
vanilla    44 training_summary_20260502_123648_energy_vanilla_s44.json 1.416874 1.190325 0.728618 0.985492 0.985091            1.964796         0.263268
vanilla    45 training_summary_20260502_123912_energy_vanilla_s45.json 1.666519 1.290937 0.816896 0.980963 0.979529            1.955413         0.276556