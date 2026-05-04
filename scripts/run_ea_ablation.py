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
    p.add_argument("--unmasked", action="store_true", help="Disable quality/fallback gates for all variants")
    p.add_argument(
        "--qerr-focus",
        action="store_true",
        help="Force mechanism-focused settings to isolate q_err effect from policy/compactness artifacts",
    )
    p.add_argument("--quiet-subprocess", action="store_true", help="Write child stdout/stderr into per-variant logs")

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
        "task_only": {
            "shap_reg.autonomous_error_shap": False,
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
            "shap_reg.gamma_consistency": 0.0,
            "shap_reg.gamma_faithfulness": 0.0,
            "shap_reg.gamma_stability": 0.0,
        },
        "full": {
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
        },
        "full_rho1": {
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
            "shap_reg.error_target_rho": 1.0,
        },
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
            "shap_reg.acceptance_min_delta_r2": -1.0,
            "shap_reg.quality_policy.reject_unstable_predictions": False,
            "shap_reg.quality_policy.mode": "quality_only",
        },
        "no_feature_gates": {
            "shap_reg.use_feature_gates": False,
        },
        "random_target": {
            "shap_reg.ea_target_ablation_mode": "random_target",
            "shap_reg.active_components": ["faithfulness"],
        },
        "shuffled_q_err": {
            "shap_reg.ea_target_ablation_mode": "shuffled_q_err",
            "shap_reg.active_components": ["faithfulness"],
        },
        "anti_q_err": {
            "shap_reg.ea_target_ablation_mode": "anti_q_err",
            "shap_reg.active_components": ["faithfulness"],
        },
        "uniform_target": {
            "shap_reg.ea_target_ablation_mode": "uniform_target",
            "shap_reg.active_components": ["faithfulness"],
        },
        "frozen_q_err": {
            "shap_reg.ea_target_ablation_mode": "frozen_q_err",
            "shap_reg.active_components": ["faithfulness"],
        },
        "no_positive_clipping": {
            "shap_reg.ea_positive_clipping": False,
            "shap_reg.active_components": ["faithfulness"],
        },
        "sparsity_only": {
            "shap_reg.autonomous_error_shap": False,
            "shap_reg.active_components": ["sparsity"],
            "shap_reg.gamma_sparsity": 1.0,
            "shap_reg.gamma_consistency": 0.0,
            "shap_reg.gamma_faithfulness": 0.0,
            "shap_reg.gamma_stability": 0.0,
        },
        "gamma_x05": {"shap_reg.gamma": 0.0005},
        "gamma_x10": {"shap_reg.gamma": 0.001},
        "gamma_x20": {"shap_reg.gamma": 0.002},
        "gamma_x03_rho1": {
            "shap_reg.gamma": 0.0003,
            "shap_reg.error_target_rho": 1.0,
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
        },
        "gamma_x10_rho1": {
            "shap_reg.gamma": 0.0010,
            "shap_reg.error_target_rho": 1.0,
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
        },
        "gamma_x30_rho1": {
            "shap_reg.gamma": 0.0030,
            "shap_reg.error_target_rho": 1.0,
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
        },
        "gamma_x100_rho1": {
            "shap_reg.gamma": 0.0100,
            "shap_reg.error_target_rho": 1.0,
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
        },
        "full_strong": {
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.gamma_sparsity": 0.0,
            "shap_reg.use_adaptive_gamma": False,
            "shap_reg.gamma": 0.05,
            "shap_reg.gamma_start": 0.05,
            "shap_reg.gamma_end": 0.05,
            "shap_reg.ea_warmup_fraction": 0.0,
            "shap_reg.ea_use_grad_balance": False,
            "shap_reg.error_target_rho": 1.0,
        },
        "random_target_strong": {
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.ea_target_ablation_mode": "random_target",
            "shap_reg.use_adaptive_gamma": False,
            "shap_reg.gamma": 0.05,
            "shap_reg.gamma_start": 0.05,
            "shap_reg.gamma_end": 0.05,
            "shap_reg.ea_warmup_fraction": 0.0,
            "shap_reg.ea_use_grad_balance": False,
            "shap_reg.error_target_rho": 1.0,
        },
        "anti_q_err_strong": {
            "shap_reg.active_components": ["faithfulness"],
            "shap_reg.ea_target_ablation_mode": "anti_q_err",
            "shap_reg.use_adaptive_gamma": False,
            "shap_reg.gamma": 0.05,
            "shap_reg.gamma_start": 0.05,
            "shap_reg.gamma_end": 0.05,
            "shap_reg.ea_warmup_fraction": 0.0,
            "shap_reg.ea_use_grad_balance": False,
            "shap_reg.error_target_rho": 1.0,
        },
        "align_js": {"shap_reg.ea_alignment_loss": "js", "shap_reg.active_components": ["faithfulness"]},
        "align_mse": {"shap_reg.ea_alignment_loss": "mse", "shap_reg.active_components": ["faithfulness"]},
        "align_cosine": {"shap_reg.ea_alignment_loss": "cosine", "shap_reg.active_components": ["faithfulness"]},
        "align_js_mse": {"shap_reg.ea_alignment_loss": "js_mse", "shap_reg.active_components": ["faithfulness"]},
        "align_cosine_mse": {"shap_reg.ea_alignment_loss": "cosine_mse", "shap_reg.active_components": ["faithfulness"]},
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


def _unmasked_patch() -> dict:
    return {
        "shap_reg.quality_first": False,
        "shap_reg.reject_on_val_degrade": False,
        "shap_reg.restore_best_state": False,
        "shap_reg.accuracy_guard.enabled": False,
        "shap_reg.acceptance_min_delta_r2": -1.0,
        "shap_reg.quality_policy.mode": "quality_only",
        "shap_reg.quality_policy.reject_unstable_predictions": False,
        "shap_reg.quality_policy.require_ea_beats_vanilla": False,
        "shap_reg.quality_policy.auc_gap_margin": 0.0,
        "shap_reg.quality_policy.min_ea_auc_gap": -1.0e9,
        "shap_reg.quality_policy.min_top_random_ratio": -1.0e9,
    }


def _qerr_focus_patch() -> dict:
    return {
        "shap_reg.autonomous_error_shap": True,
        "shap_reg.active_components": ["faithfulness"],
        "shap_reg.use_adaptive_gamma": False,
        "shap_reg.gamma": 0.05,
        "shap_reg.gamma_start": 0.05,
        "shap_reg.gamma_end": 0.05,
        "shap_reg.gamma_warmup_epochs": 0.0,
        "shap_reg.gamma_sparsity": 0.0,
        "shap_reg.gamma_consistency": 0.0,
        "shap_reg.gamma_faithfulness": 0.0,
        "shap_reg.gamma_stability": 0.0,
        "shap_reg.use_feature_gates": False,
        "shap_reg.ea_importance_source": "grad",
        "shap_reg.error_target_rho": 1.0,
        "shap_reg.ea_alignment_loss": "js_mse",
        "shap_reg.ea_alignment_alpha": 0.5,
        "shap_reg.ea_warmup_fraction": 0.0,
        "shap_reg.ea_use_grad_balance": False,
        "shap_reg.error_js_weight": 1.0,
        "shap_reg.error_mse_weight": 0.0,
        "shap_reg.gate_js_weight": 0.0,
        "shap_reg.noise_weight": 0.0,
        "shap_reg.rule_stability_weight": 0.0,
        "shap_reg.ea_bottom_invariance_weight": 0.0,
        "shap_reg.ea_rank_weight": 0.0,
        "shap_reg.ea_positive_clipping": True,
    }


def _run(cmd: list[str], log_path: Path | None = None):
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    if args.quiet_subprocess:
        logs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.base_config).resolve()
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    variants_def = _default_variants()
    alias = {
        "neg_core_v2": [
            "full_rho1",
            "random_target",
            "shuffled_q_err",
            "uniform_target",
            "anti_q_err",
            "sparsity_only",
            "task_only",
        ],
        "gamma_sweep_v2": [
            "gamma_x03_rho1",
            "gamma_x10_rho1",
            "gamma_x30_rho1",
            "gamma_x100_rho1",
        ],
    }
    varg = args.variants.strip().lower()
    if varg == "default":
        variants_req = list(variants_def.keys())
    elif varg in alias:
        variants_req = alias[varg]
    else:
        variants_req = [v.strip() for v in args.variants.split(",") if v.strip()]

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
        cfg_v = deepcopy(base_cfg)
        if args.qerr_focus:
            cfg_v = _apply_patch(cfg_v, _qerr_focus_patch())
        cfg_v = _apply_patch(cfg_v, patch)
        if args.unmasked:
            cfg_v = _apply_patch(cfg_v, _unmasked_patch())
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
        var_log = (logs_dir / f"{vname}.log") if args.quiet_subprocess else None
        _run(cmd, var_log)

        multiseed_path = Path("results") / f"multiseed_{cfg_path.stem}_{tag_prefix}.json"
        var_row = {
            "config": str(cfg_path.resolve()),
            "patch": patch,
            "multiseed": str(multiseed_path.resolve()),
        }
        if args.quiet_subprocess:
            var_row["log"] = str((logs_dir / f"{vname}.log").resolve())

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
            _run(cmd_exp, var_log)
            var_row["explainability"] = str(out_exp.resolve())

        manifest["variants"][vname] = var_row

    manifest_path = out_dir / f"ablation_manifest_{base_cfg_path.stem}_{args.tag_prefix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()
