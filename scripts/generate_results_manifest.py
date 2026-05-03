#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from datetime import datetime


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    def pick_existing(candidates):
        for c in candidates:
            if Path(c).exists():
                return c
        return candidates[0]

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "tables": {
            "r2_multidataset": "results/methods_compare_multidataset_20260503.md",
            "faithfulness_ea_vs_vanilla": "results/sml2010_ea_minimal_vs_vanilla_faithfulness.md",
            "faithfulness_baselines_vs_ea": "results/faithfulness_baselines_vs_ea_20260503.md",
            "mlp_portability_sml2010": pick_existing([
                "results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md",
            ]),
            "faithfulness_mode_compare": pick_existing([
                "results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md",
                "results/sml2010_faithfulness_mode_compare_20260503.md",
            ]),
            "ablation_nf3_ea_raw": pick_existing([
                "results/ablation_nf3_ea_raw_summary_20260503.md",
                "results/ablation_summary_ablation_manifest_config_sml2010_ea_minimal_sml_ablation_neg3_fast.md",
            ]),
            "stability_sensitivity": pick_existing([
                "results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.md",
                "results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_ea5_full_20260503.md",
                "results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_guard_smoke3_20260503.md",
            ]),
        },
        "artifacts": {
            "sml_multiseed": "results/multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt.json",
            "energy_multiseed": "results/multiseed_config_energy_ea_minimal_energy_ea10.json",
            "naval_multiseed": "results/multiseed_config_naval_ea_minimal_naval_ea_diag10.json",
            "sml_nonfast_5": pick_existing([
                "results/multiseed_config_sml2010_ea_minimal_sml_ea5_full.json",
            ]),
            "sml_nonfast_5_stability_guard": pick_existing([
                "results/multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5.json",
            ]),
            "mlp_multiseed_sml2010_5": pick_existing([
                "results/mlp_eaar_multiseed_config_sml2010_mlp_ea_sml_mlp_eaar5.json",
            ]),
            "baseline_sml": "results/baselines_sml2010_sml2010_10seed.json",
            "baseline_energy": "results/baselines_energy_efficiency_energy_10seed.json",
            "baseline_naval": "results/baselines_naval_propulsion_naval_10seed.json",
            "significance_auc_gap_sml": pick_existing([
                "results/significance_sml2010_ea_vs_vanilla_auc_gap.json",
            ]),
        },
        "configs": {
            "ea_sml": "configs/config_sml2010_ea_minimal.yaml",
            "ea_energy": "configs/config_energy_ea_minimal.yaml",
            "ea_naval": "configs/config_naval_ea_minimal.yaml",
            "vanilla_sml": "configs/config_sml2010_vanilla_real_only.yaml",
            "vanilla_energy": "configs/config_energy_vanilla_real_only.yaml",
            "vanilla_naval": "configs/config_naval_vanilla_real_only.yaml",
        },
    }
    out = Path("results/results_manifest.json")
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
