"""Prediction diagnostics and unstable-run flags for regression outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    means = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - means) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    return r2, ss_res, ss_tot


def compute_prediction_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train_ref: np.ndarray | None = None,
    min_r2: float = 0.0,
    max_abs_error_p99_ratio: float = 10.0,
    max_abs_error_target_std_ratio: float = 20.0,
    max_prediction_range_std_ratio: float = 5.0,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    if y_true.shape != y_pred.shape:
        m = min(y_true.shape[0], y_pred.shape[0])
        d = min(y_true.shape[1], y_pred.shape[1])
        y_true = y_true[:m, :d]
        y_pred = y_pred[:m, :d]

    nan_inf_flag = (not np.isfinite(y_true).all()) or (not np.isfinite(y_pred).all())
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    err = y_true - y_pred
    abs_err = np.abs(err)
    abs_err_p95 = float(np.quantile(abs_err, 0.95))
    abs_err_p99 = float(np.quantile(abs_err, 0.99))
    abs_err_max = float(np.max(abs_err))

    y_std_per_target = np.std(y_true, axis=0)
    target_std_mean = float(np.mean(y_std_per_target)) if y_std_per_target.size else 0.0
    target_std_mean = max(target_std_mean, 1e-12)

    r2_vec, ss_res_vec, ss_tot_vec = _r2_per_target(y_true, y_pred)
    r2_mean = float(np.mean(r2_vec)) if r2_vec.size else float("nan")
    negative_r2_flag = r2_mean < _safe_float(min_r2, 0.0)

    extreme_error_flag = abs_err_max > max(
        _safe_float(max_abs_error_p99_ratio, 10.0) * max(abs_err_p99, 1e-12),
        _safe_float(max_abs_error_target_std_ratio, 20.0) * target_std_mean,
    )

    if y_train_ref is not None:
        y_ref = np.asarray(y_train_ref, dtype=float)
        if y_ref.ndim == 1:
            y_ref = y_ref[:, None]
        y_ref = np.nan_to_num(y_ref, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        y_ref = y_true
    y_ref_min = float(np.min(y_ref))
    y_ref_max = float(np.max(y_ref))
    y_ref_std = float(np.std(y_ref))
    y_ref_std = max(y_ref_std, 1e-12)
    pred_min = float(np.min(y_pred))
    pred_max = float(np.max(y_pred))
    range_margin = _safe_float(max_prediction_range_std_ratio, 5.0) * y_ref_std
    prediction_range_flag = (pred_min < (y_ref_min - range_margin)) or (pred_max > (y_ref_max + range_margin))

    unstable_prediction_flag = bool(
        nan_inf_flag or negative_r2_flag or extreme_error_flag or prediction_range_flag
    )

    per_target = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        ej = yt - yp
        per_target.append(
            {
                "target_idx": int(j),
                "r2": float(r2_vec[j]),
                "mse": float(np.mean(ej * ej)),
                "mae": float(np.mean(np.abs(ej))),
                "rmse": float(np.sqrt(np.mean(ej * ej))),
                "ss_res": float(ss_res_vec[j]),
                "ss_tot": float(ss_tot_vec[j]),
                "y_mean": float(np.mean(yt)),
                "y_std": float(np.std(yt)),
                "y_min": float(np.min(yt)),
                "y_max": float(np.max(yt)),
                "pred_mean": float(np.mean(yp)),
                "pred_std": float(np.std(yp)),
                "pred_min": float(np.min(yp)),
                "pred_max": float(np.max(yp)),
                "abs_err_p95": float(np.quantile(np.abs(ej), 0.95)),
                "abs_err_p99": float(np.quantile(np.abs(ej), 0.99)),
                "abs_err_max": float(np.max(np.abs(ej))),
            }
        )

    return {
        "unstable_prediction_flag": unstable_prediction_flag,
        "nan_inf_flag": bool(nan_inf_flag),
        "negative_r2_flag": bool(negative_r2_flag),
        "extreme_error_flag": bool(extreme_error_flag),
        "prediction_range_flag": bool(prediction_range_flag),
        "r2_mean": float(r2_mean),
        "abs_err_p95": abs_err_p95,
        "abs_err_p99": abs_err_p99,
        "abs_err_max": abs_err_max,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "target_std_mean": float(target_std_mean),
        "train_target_min": y_ref_min,
        "train_target_max": y_ref_max,
        "train_target_std": y_ref_std,
        "thresholds": {
            "min_r2": _safe_float(min_r2, 0.0),
            "max_abs_error_p99_ratio": _safe_float(max_abs_error_p99_ratio, 10.0),
            "max_abs_error_target_std_ratio": _safe_float(max_abs_error_target_std_ratio, 20.0),
            "max_prediction_range_std_ratio": _safe_float(max_prediction_range_std_ratio, 5.0),
        },
        "per_target": per_target,
    }

