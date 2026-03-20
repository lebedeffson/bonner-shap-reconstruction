#!/usr/bin/env python3
"""Небольшой воспроизводимый тюнинг V2 против текущего V1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
BASELINE_SUMMARY = REPO_ROOT / "results" / "training_summary_20260319_190809_tikhonov_stronger_20260319.json"
BASE_CONFIG = REPO_ROOT / "configs" / "config_integrated_shap_v2.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def better_than_baseline(candidate_metrics: dict, baseline_metrics: dict) -> bool:
    return (
        candidate_metrics["mse"] < baseline_metrics["mse"]
        and candidate_metrics["rmse"] < baseline_metrics["rmse"]
        and candidate_metrics["mae"] < baseline_metrics["mae"]
        and candidate_metrics["r2_weighted"] > baseline_metrics["r2_weighted"]
        and candidate_metrics["r2_mean"] > baseline_metrics["r2_mean"]
    )


def metric_deltas(candidate_metrics: dict, baseline_metrics: dict) -> dict:
    return {
        "delta_mse": candidate_metrics["mse"] - baseline_metrics["mse"],
        "delta_rmse": candidate_metrics["rmse"] - baseline_metrics["rmse"],
        "delta_mae": candidate_metrics["mae"] - baseline_metrics["mae"],
        "delta_r2_weighted": candidate_metrics["r2_weighted"] - baseline_metrics["r2_weighted"],
        "delta_r2_mean": candidate_metrics["r2_mean"] - baseline_metrics["r2_mean"],
    }


def candidate_definitions(profile: str) -> list[dict]:
    if profile == "from_v1":
        return [
            {
                "name": "v2_eqbands_tikh0012_nonneg005",
                "tikhonov_lambda": 0.0012,
                "nonneg_lambda": 0.005,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2_eqbands_tikh0010_nonneg005",
                "tikhonov_lambda": 0.0010,
                "nonneg_lambda": 0.005,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2_eqbands_tikh0010_nonneg003",
                "tikhonov_lambda": 0.0010,
                "nonneg_lambda": 0.003,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2_eqbands_tikh0012_nonneg003",
                "tikhonov_lambda": 0.0012,
                "nonneg_lambda": 0.003,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2_midband_tikh0012_nonneg005",
                "tikhonov_lambda": 0.0012,
                "nonneg_lambda": 0.005,
                "band_weights": [0.25, 0.50, 0.25],
                "target_shap_ratio": 0.35,
                "gamma_end": 0.09,
            },
        ]

    if profile == "next_step":
        return [
            {
                "name": "v3_eqbands_tikh0009_nonneg005",
                "tikhonov_lambda": 0.0009,
                "nonneg_lambda": 0.005,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v3_eqbands_tikh0010_nonneg004",
                "tikhonov_lambda": 0.0010,
                "nonneg_lambda": 0.004,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v3_eqbands_tikh0009_nonneg0045",
                "tikhonov_lambda": 0.0009,
                "nonneg_lambda": 0.0045,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.42,
                "gamma_end": 0.10,
            },
            {
                "name": "v3_eqbands_tikh0011_nonneg004",
                "tikhonov_lambda": 0.0011,
                "nonneg_lambda": 0.004,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.42,
                "gamma_end": 0.10,
            },
            {
                "name": "v3_eqbands_tikh0009_nonneg004_gamma011",
                "tikhonov_lambda": 0.0009,
                "nonneg_lambda": 0.004,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.42,
                "gamma_end": 0.11,
            },
        ]

    if profile == "scheduled_step":
        return [
            {
                "name": "v2sched_eq_tikh0010_nonneg005_warm60",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0010,
                "tikhonov_warmup_epochs": 0.0,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2sched_eq_tikh0005to0010_nonneg005_warm60",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2sched_eq_tikh0007to0010_nonneg0045_warm50",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0007,
                "tikhonov_warmup_epochs": 0.30,
                "nonneg_lambda": 0.0045,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.50,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2sched_hiband_tikh0005to0010_nonneg0045",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0045,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.55,
                "band_weights": [0.30, 0.30, 0.40],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2sched_hiband_tikh0005to0010_nonneg005_ratio038",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.30, 0.30, 0.40],
                "target_shap_ratio": 0.38,
                "gamma_end": 0.10,
            },
        ]

    if profile == "micro_step":
        return [
            {
                "name": "v2micro_eq_sched_nonneg0048",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0048,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2micro_eq_sched_target039",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2micro_eq_sched_gamma0095",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.095,
            },
            {
                "name": "v2micro_soft_hiband_nonneg0048",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0048,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
            {
                "name": "v2micro_soft_hiband_target039",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2micro_midhiband_nonneg0048",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0048,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.31, 0.34, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2micro_eq_sched_tikh00045to00095_nonneg0048",
                "tikhonov_lambda": 0.00095,
                "tikhonov_lambda_start": 0.00045,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0048,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.40,
                "gamma_end": 0.10,
            },
        ]

    if profile == "margin_step":
        return [
            {
                "name": "v2margin_eq_target039_tol0010",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "nonneg_mode": "margin_mass_ratio",
                "nonneg_tolerance": 0.010,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2margin_eq_target039_tol0015",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "nonneg_mode": "margin_mass_ratio",
                "nonneg_tolerance": 0.015,
                "band_weights": [1 / 3, 1 / 3, 1 / 3],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2margin_soft_hiband_target039_tol0010",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "nonneg_mode": "margin_mass_ratio",
                "nonneg_tolerance": 0.010,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2margin_soft_hiband_target039_tol0015",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "nonneg_mode": "margin_mass_ratio",
                "nonneg_tolerance": 0.015,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2margin_midhiband_target039_tol0010",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "nonneg_mode": "margin_mass_ratio",
                "nonneg_tolerance": 0.010,
                "band_weights": [0.31, 0.34, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
        ]

    if profile == "blend_step":
        return [
            {
                "name": "v2blend_soft_hiband_target039",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.10,
            },
            {
                "name": "v2blend_soft_hiband_target039_gamma0095",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.32, 0.33, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.095,
            },
            {
                "name": "v2blend_gentle_hiband_target039_gamma0095",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.325, 0.335, 0.34],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.095,
            },
            {
                "name": "v2blend_gentle_hiband_target0385_gamma0095",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.325, 0.335, 0.34],
                "target_shap_ratio": 0.385,
                "gamma_end": 0.095,
            },
            {
                "name": "v2blend_midhiband_target039_gamma0095",
                "tikhonov_lambda": 0.0010,
                "tikhonov_lambda_start": 0.0005,
                "tikhonov_warmup_epochs": 0.35,
                "nonneg_lambda": 0.0050,
                "nonneg_lambda_start": 0.0,
                "nonneg_warmup_epochs": 0.60,
                "band_weights": [0.31, 0.34, 0.35],
                "target_shap_ratio": 0.39,
                "gamma_end": 0.095,
            },
        ]

    raise ValueError(f"Unknown search profile: {profile}")


def build_candidate_config(base_config: dict, candidate: dict, output_dir: Path) -> dict:
    cfg = deepcopy(base_config)
    shap = cfg["shap_reg"]
    shap["target_shap_ratio"] = float(candidate["target_shap_ratio"])
    shap["gamma"] = float(candidate["gamma_end"])
    shap["gamma_end"] = float(candidate["gamma_end"])
    shap["scalarization"]["mode"] = "band_weighted"
    shap["scalarization"]["band_weights"] = [float(x) for x in candidate["band_weights"]]
    shap["tikhonov"]["enabled"] = True
    shap["tikhonov"]["energy_aware"] = True
    shap["tikhonov"]["lambda"] = float(candidate["tikhonov_lambda"])
    shap["tikhonov"]["lambda_start"] = float(candidate.get("tikhonov_lambda_start", candidate["tikhonov_lambda"]))
    shap["tikhonov"]["lambda_end"] = float(candidate.get("tikhonov_lambda", candidate["tikhonov_lambda"]))
    shap["tikhonov"]["warmup_epochs"] = float(candidate.get("tikhonov_warmup_epochs", 0.0))
    shap["nonnegativity"]["enabled"] = True
    shap["nonnegativity"]["mode"] = str(candidate.get("nonneg_mode", "mass_ratio"))
    shap["nonnegativity"]["lambda"] = float(candidate["nonneg_lambda"])
    shap["nonnegativity"]["lambda_start"] = float(candidate.get("nonneg_lambda_start", candidate["nonneg_lambda"]))
    shap["nonnegativity"]["lambda_end"] = float(candidate.get("nonneg_lambda", candidate["nonneg_lambda"]))
    shap["nonnegativity"]["warmup_epochs"] = float(candidate.get("nonneg_warmup_epochs", 0.0))
    if "nonneg_tolerance" in candidate:
        shap["nonnegativity"]["tolerance"] = float(candidate["nonneg_tolerance"])
    cfg["output"]["results_dir"] = str(output_dir)
    cfg["output"]["save_model"] = False
    cfg["output"]["save_predictions"] = False
    cfg["output"]["save_plots"] = False
    cfg["output"]["save_samples"] = False
    return cfg


def run_candidate(config_path: Path, tag: str) -> Path:
    cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(config_path),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    result_dir = config_path.parent
    candidates = sorted(result_dir.glob(f"training_summary_*_{tag}.json"))
    if not candidates:
        raise FileNotFoundError(f"Не найден summary для tag={tag} в {result_dir}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Подбор V2-конфигов против V1 baseline")
    parser.add_argument("--baseline-summary", default=str(BASELINE_SUMMARY))
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / f"v2_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--stop-on-success", action="store_true", help="Остановиться на первом кандидате, который лучше baseline по всем метрикам")
    parser.add_argument(
        "--search-profile",
        default="from_v1",
        choices=["from_v1", "next_step", "scheduled_step", "micro_step", "margin_step", "blend_step"],
    )
    args = parser.parse_args()

    baseline_summary = load_summary(Path(args.baseline_summary))
    baseline_metrics = baseline_summary["metrics"]
    base_config = load_yaml(Path(args.base_config))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    winner = None

    for candidate in candidate_definitions(args.search_profile):
        candidate_dir = output_dir / candidate["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        config_path = candidate_dir / f"{candidate['name']}.yaml"
        dump_yaml(config_path, build_candidate_config(base_config, candidate, candidate_dir))

        summary_path = run_candidate(config_path, candidate["name"])
        summary = load_summary(summary_path)
        metrics = summary["metrics"]
        deltas = metric_deltas(metrics, baseline_metrics)
        success = better_than_baseline(metrics, baseline_metrics)

        row = {
            "name": candidate["name"],
            "summary_path": str(summary_path),
            **candidate,
            **metrics,
            **deltas,
            "success": success,
            "negative_fraction": summary.get("diagnostics", {}).get("prediction_stats", {}).get("negative_fraction"),
        }
        rows.append(row)

        if success and winner is None:
            winner = row
            if args.stop_on_success:
                break

    csv_path = output_dir / "v2_tuning_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "v2_tuning_summary.md"
    lines = [
        "# V2 Tuning Summary",
        "",
        f"- Baseline summary: `{args.baseline_summary}`",
        f"- Output CSV: `{csv_path}`",
        "",
        "| name | mse | rmse | mae | r2_weighted | r2_mean | success | negative_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['mse']:.8f} | {row['rmse']:.8f} | {row['mae']:.8f} | "
            f"{row['r2_weighted']:.8f} | {row['r2_mean']:.8f} | {str(row['success'])} | {row['negative_fraction']:.6f} |"
        )
    if winner:
        lines.extend(
            [
                "",
                "## Winner",
                "",
                f"- {winner['name']}",
                f"- summary: `{winner['summary_path']}`",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved tuning summary: {csv_path}")
    if winner:
        print(f"Winner: {winner['name']}")
        return 0
    print("No candidate beat baseline on all metrics.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
