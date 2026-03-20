"""
Precision-aware SHAP regularization utilities.
Provides stable, differentiable components for consistency, sparsity,
faithfulness, stability, and adaptive component weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PrecisionOptimizedSHAPRegularization:
    """Static helpers for precision-aware SHAP regularization."""

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        vec = torch.clamp(vec, min=eps)
        return vec / (torch.sum(vec) + eps)

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        p = PrecisionOptimizedSHAPRegularization._safe_normalize(p, eps=eps)
        q = PrecisionOptimizedSHAPRegularization._safe_normalize(q, eps=eps)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
        return 0.5 * (kl_pm + kl_qm)

    @staticmethod
    def compute_precision_aware_consistency(
        grad_importance_normalized: torch.Tensor,
        true_shap_normalized: torch.Tensor,
        current_main_loss: float,
        use_adaptive: bool = True,
        js_weight: float = 0.5,
    ) -> dict:
        """Consistency = MSE + JS divergence, scaled by model precision."""
        eps = 1e-10
        p = PrecisionOptimizedSHAPRegularization._safe_normalize(grad_importance_normalized, eps=eps)
        q = PrecisionOptimizedSHAPRegularization._safe_normalize(true_shap_normalized, eps=eps)

        mse_loss = torch.mean((p - q) ** 2)
        js_loss = PrecisionOptimizedSHAPRegularization._js_divergence(p, q, eps=eps)
        consistency_loss = mse_loss + js_weight * js_loss

        if use_adaptive:
            scale = 1.0 / (1.0 + float(current_main_loss))
            consistency_loss = consistency_loss * scale

        return {
            "consistency_loss": consistency_loss,
            "mse_loss": mse_loss,
            "js_loss": js_loss,
        }

    @staticmethod
    def _gini_coefficient(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        p = PrecisionOptimizedSHAPRegularization._safe_normalize(p, eps=eps)
        sorted_p, _ = torch.sort(p)
        n = sorted_p.numel()
        if n <= 1:
            return torch.tensor(0.0, device=p.device, dtype=p.dtype)
        idx = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
        gini = (2.0 * torch.sum(idx * sorted_p) / (n * torch.sum(sorted_p) + eps)) - (n + 1) / n
        return torch.clamp(gini, min=0.0, max=1.0)

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        p = PrecisionOptimizedSHAPRegularization._safe_normalize(p, eps=eps)
        n = p.numel()
        if n <= 1:
            return torch.tensor(0.0, device=p.device, dtype=p.dtype)
        entropy = -torch.sum(p * torch.log(p + eps))
        return entropy / (torch.log(torch.tensor(float(n), device=p.device, dtype=p.dtype)) + eps)

    @staticmethod
    def compute_precision_aware_sparsity(
        grad_importance_normalized: torch.Tensor,
        current_main_loss: float,
        target_gini: float = 0.3,
        precision_weight: float = 0.7,
    ) -> dict:
        """Sparsity = entropy + gini deviation, scaled by precision."""
        eps = 1e-10
        p = PrecisionOptimizedSHAPRegularization._safe_normalize(grad_importance_normalized, eps=eps)

        gini = PrecisionOptimizedSHAPRegularization._gini_coefficient(p, eps=eps)
        entropy = PrecisionOptimizedSHAPRegularization._entropy(p, eps=eps)

        target_gini_t = torch.tensor(float(target_gini), device=p.device, dtype=p.dtype)
        gini_loss = torch.relu(target_gini_t - gini) ** 2

        base_loss = entropy + gini_loss
        precision_factor = 1.0 / (1.0 + float(current_main_loss))
        scale = precision_weight * precision_factor + (1.0 - precision_weight)
        sparsity_loss = base_loss * scale

        return {
            "sparsity_loss": sparsity_loss,
            "gini_coefficient": gini,
            "gini_loss": gini_loss,
            "entropy": entropy,
        }

    @staticmethod
    def compute_precision_aware_faithfulness(
        batch_X: torch.Tensor,
        baseline_X: torch.Tensor,
        predictions: torch.Tensor,
        baseline_pred: torch.Tensor,
        model,
        current_main_loss: float,
        order: int = 1,
        scalarize_fn=None,
    ) -> dict:
        """
        Faithfulness: compare true output change with first-order approximation.
        Uses mean output over bins for stability.
        """
        if order != 1:
            order = 1

        # Scalarized outputs for a stable, comparable signal
        if scalarize_fn is None:
            pred_mean = torch.mean(predictions, dim=1)
            baseline_mean = torch.mean(baseline_pred, dim=1)
        else:
            pred_mean = scalarize_fn(predictions)
            baseline_mean = scalarize_fn(baseline_pred)

        grad_input = torch.autograd.grad(
            outputs=pred_mean,
            inputs=batch_X,
            grad_outputs=torch.ones_like(pred_mean),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        approx = torch.sum(grad_input * (batch_X - baseline_X), dim=1)
        delta = pred_mean - baseline_mean
        faithfulness_loss = torch.mean((delta - approx) ** 2)

        scale = 1.0 / (1.0 + float(current_main_loss))
        faithfulness_loss = faithfulness_loss * scale

        return {"faithfulness_loss": faithfulness_loss}

    @staticmethod
    def compute_precision_aware_stability(
        importance_per_sample: torch.Tensor,
        current_main_loss: float,
    ) -> dict:
        """Stability = variance of importance across samples."""
        if importance_per_sample.ndim == 1:
            stability_loss = torch.tensor(0.0, device=importance_per_sample.device)
        else:
            mean_imp = torch.mean(importance_per_sample, dim=0, keepdim=True)
            stability_loss = torch.mean((importance_per_sample - mean_imp) ** 2)

        scale = 1.0 / (1.0 + float(current_main_loss))
        stability_loss = stability_loss * scale
        return {"stability_loss": stability_loss}

    @staticmethod
    def compute_adaptive_component_weights(
        current_main_loss: float,
        consistency_loss: float,
        sparsity_loss: float,
        faithfulness_loss: float,
        stability_loss: float,
        target_main_loss: float = 0.02,
    ) -> dict:
        """Adaptive weights based on inverse loss magnitude."""
        eps = 1e-8
        losses = np.array(
            [consistency_loss, sparsity_loss, faithfulness_loss, stability_loss],
            dtype=float,
        )
        losses = np.where(np.isfinite(losses), losses, np.nan)
        losses = np.nan_to_num(losses, nan=1.0, posinf=1.0, neginf=1.0)
        losses = np.maximum(losses, eps)

        inv = 1.0 / losses
        weights = inv / np.sum(inv)

        if current_main_loss > target_main_loss:
            scale = target_main_loss / (current_main_loss + eps)
            weights = weights * np.array([1.0, 1.0, scale, scale], dtype=float)
            weights = weights / np.sum(weights)

        return {
            "weights": {
                "consistency": float(weights[0]),
                "sparsity": float(weights[1]),
                "faithfulness": float(weights[2]),
                "stability": float(weights[3]),
            }
        }
