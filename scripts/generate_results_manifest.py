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
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "tables": {
            "r2_multidataset": "results/methods_compare_multidataset_20260503.md",
            "faithfulness_ea_vs_vanilla": "results/sml2010_ea_minimal_vs_vanilla_faithfulness.md",
            "faithfulness_baselines_vs_ea": "results/faithfulness_baselines_vs_ea_20260503.md",
        },
        "artifacts": {
            "sml_multiseed": "results/multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt.json",
            "energy_multiseed": "results/multiseed_config_energy_ea_minimal_energy_ea10.json",
            "naval_multiseed": "results/multiseed_config_naval_ea_minimal_naval_ea_diag10.json",
            "baseline_sml": "results/baselines_sml2010_sml2010_10seed.json",
            "baseline_energy": "results/baselines_energy_efficiency_energy_10seed.json",
            "baseline_naval": "results/baselines_naval_propulsion_naval_10seed.json",
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

