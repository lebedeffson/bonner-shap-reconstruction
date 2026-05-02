"""Диагностика качества признаков для регрессионных задач."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.apply(pd.to_numeric, errors="coerce")


def detect_constant_features(X: pd.DataFrame) -> list[str]:
    df = _numeric_df(X)
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def detect_duplicate_features(X: pd.DataFrame) -> list[tuple[str, str]]:
    df = _numeric_df(X)
    duplicates: list[tuple[str, str]] = []
    seen: list[str] = []
    for col in df.columns:
        matched = None
        for ref in seen:
            if df[col].equals(df[ref]):
                matched = ref
                break
        if matched is None:
            seen.append(col)
        else:
            duplicates.append((matched, col))
    return duplicates


def detect_high_correlation_pairs(
    X: pd.DataFrame,
    threshold: float = 0.9999,
    max_pairs: int = 100,
) -> list[tuple[str, str, float]]:
    df = _numeric_df(X).dropna(axis=1, how="all")
    if df.shape[1] < 2:
        return []
    corr = df.corr().abs()
    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            val = float(corr.loc[left, right])
            if np.isfinite(val) and val >= threshold:
                pairs.append((left, right, val))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def estimate_condition_number(X: pd.DataFrame) -> float:
    df = _numeric_df(X).dropna(axis=1, how="all").fillna(0.0)
    if df.shape[1] == 0:
        return float("nan")
    mat = df.to_numpy(dtype=float)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    mat = mat / std
    xtx = mat.T @ mat
    try:
        return float(np.linalg.cond(xtx))
    except np.linalg.LinAlgError:
        return float("inf")


def summarize_feature_quality(
    X: pd.DataFrame,
    corr_threshold: float = 0.9999,
    max_pairs: int = 20,
) -> dict[str, Any]:
    constants = detect_constant_features(X)
    duplicates = detect_duplicate_features(X)
    high_corr = detect_high_correlation_pairs(X, threshold=corr_threshold, max_pairs=max_pairs)
    cond = estimate_condition_number(X)
    return {
        "n_features": int(X.shape[1]),
        "constant_features": constants,
        "duplicate_pairs": duplicates,
        "high_corr_pairs": high_corr,
        "condition_number": cond,
    }
