#!/usr/bin/env python3
"""Run EA ablation matrix for ANFIS configs via run_multiseed_autonomous.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True, help="Base ANFIS config yaml")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--python", default="/home/lebedeffson/Code/venv_cuda/bin/python")
    p.add_argument("--tag-prefix", default="ablation")
    p.add_argument("--variants", default="default", help="comma list or 'default'")
    p.add_argument("--out-dir", default="results/ablation")
    p.add_argument("--with-explainability", action="store_true")
    p.add_argument("--k-list", default="1,2,3,4")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument(
        "--eval-importance",
        choices=[
            "final",
            "shap",
            "ea_raw",
            "ea-only",
            "shap-only",
            "vanilla",
            "vanilla_gradient",
            "vanilla_permutation",
        ],
        default="final",
    )
    p.add_argument("--random-trials", type=int, default=20)

    # passthrough for run_multiseed_autonomous
    p.add_argument("--inprocess", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--pso-epochs", type=int, default=25)
    p.add_argument("--pso-pop", type=int, default=30)
    p.add_argument("--shap-epochs", type=int, default=15)
    p.add_argument("--fast-save-model", action="store_true")
    return p.parse_args()


def _default_variants():
    return {
        "full": {},
        "eaar_bottom_001": {
            "shap_reg.ea_bottom_invariance_weight": 0.01,
            "shap_reg.ea_importance_source": "mixed",
            "shap_reg.ea_gate_importance_alpha": 0.5,
        },
        "eaar_bottom_003": {
            "shap_reg.ea_bottom_invariance_weight": 0.03,
            "shap_reg.ea_importance_source": "mixed",
            "shap_reg.ea_gate_importance_alpha": 0.5,
        },
        "eaar_bottom_007": {
            "shap_reg.ea_bottom_invariance_weight": 0.07,
            "shap_reg.ea_importance_source": "mixed",
            "shap_reg.ea_gate_importance_alpha": 0.5,
        },
        "eaar_gate": {
            "shap_reg.ea_importance_source": "gate",
            "shap_reg.ea_bottom_invariance_weight": 0.03,
        },
        "eaar_mixed": {
            "shap_reg.ea_importance_source": "mixed",
            "shap_reg.ea_gate_importance_alpha": 0.5,
            "shap_reg.ea_bottom_invariance_weight": 0.03,
        },
        "no_ema": {
            "shap_reg.error_importance_ema_beta": 0.0,
            "shap_reg.grad_importance_ema_beta": 0.0,
        },
        "no_warmup": {
            "shap_reg.ea_warmup_fraction": 0.0,
            "shap_reg.gamma_warmup_epochs": 0.0,
        },
        "no_grad_balance": {
            "shap_reg.ea_use_grad_balance": False,
        },
        "train_target": {
            "shap_reg.error_importance_target": "train",
        },
        "val_target": {
            "shap_reg.error_importance_target": "val",
        },
        "mask_mean": {
            "shap_reg.error_importance_mode": "mean",
        },
        "mask_noise": {
            "shap_reg.error_importance_mode": "noise",
        },
        "no_fallback": {
            "shap_reg.quality_first": False,
            "shap_reg.reject_on_val_degrade": False,
            "shap_reg.restore_best_state": False,
            "shap_reg.accuracy_guard.enabled": False,
        },
        "no_feature_gates": {
            "shap_reg.use_feature_gates": False,
        },
        "random_target": {
            "shap_reg.ea_target_ablation_mode": "random_target",
        },
        "shuffled_q_err": {
            "shap_reg.ea_target_ablation_mode": "shuffled_q_err",
        },
        "sparsity_only": {
            "shap_reg.autonomous_error_shap": False,
            "shap_reg.active_components": ["sparsity"],
            "shap_reg.gamma_sparsity": 1.0,
            "shap_reg.gamma_consistency": 0.0,
            "shap_reg.gamma_faithfulness": 0.0,
            "shap_reg.gamma_stability": 0.0,
        },
    }


def _set_path(d: dict, path: str, value):
    parts = path.split(".")
    cur = d
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _apply_patch(cfg: dict, patch: dict):
    out = deepcopy(cfg)
    for k, v in patch.items():
        _set_path(out, k, v)
    return out


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.base_config).resolve()
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    variants_def = _default_variants()
    variants_req = (
        list(variants_def.keys()) if args.variants.strip().lower() == "default"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(base_cfg_path),
        "seeds": args.seeds,
        "variants": {},
    }

    for vname in variants_req:
        if vname not in variants_def:
            raise ValueError(f"Unknown variant: {vname}")
        patch = variants_def[vname]
        cfg_v = _apply_patch(base_cfg, patch)
        cfg_path = cfg_dir / f"{base_cfg_path.stem}_{vname}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_v, sort_keys=False, allow_unicode=True), encoding="utf-8")

        tag_prefix = f"{args.tag_prefix}_{vname}"
        cmd = [
            args.python, "scripts/run_multiseed_autonomous.py",
            "--config", str(cfg_path),
            "--seeds", args.seeds,
            "--tag-prefix", tag_prefix,
            "--python", args.python,
        ]
        if args.inprocess:
            cmd.append("--inprocess")
        if args.fast:
            cmd += [
                "--fast",
                "--pso-epochs", str(args.pso_epochs),
                "--pso-pop", str(args.pso_pop),
                "--shap-epochs", str(args.shap_epochs),
            ]
            if args.fast_save_model:
                cmd.append("--fast-save-model")
        _run(cmd)

        multiseed_path = Path("results") / f"multiseed_{cfg_path.stem}_{tag_prefix}.json"
        var_row = {
            "config": str(cfg_path.resolve()),
            "patch": patch,
            "multiseed": str(multiseed_path.resolve()),
        }

        if args.with_explainability:
            out_exp = Path("results") / (
                f"explainability_{multiseed_path.stem}_{args.mask}_{args.eval_importance}.json"
            )
            cmd_exp = [
                args.python, "scripts/report_explainability_multiseed.py",
                "--multiseed", str(multiseed_path),
                "--k-list", args.k_list,
                "--mask", args.mask,
                "--random-trials", str(args.random_trials),
                "--eval-importance", args.eval_importance,
                "--out", str(out_exp),
            ]
            _run(cmd_exp)
            var_row["explainability"] = str(out_exp.resolve())

        manifest["variants"][vname] = var_row

    manifest_path = out_dir / f"ablation_manifest_{base_cfg_path.stem}_{args.tag_prefix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()
