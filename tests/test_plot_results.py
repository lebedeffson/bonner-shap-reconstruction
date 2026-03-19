import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_results import _load_plot_metadata, plot_regularization_summary, plot_shap_history


def test_load_plot_metadata_reads_feature_and_target_names(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  feature_prefix: S",
                "  feature_count: 3",
                "  feature_index_start: 2",
                "  target_prefix: E",
                "  target_count: 4",
                "  target_index_start: 1",
            ]
        ),
        encoding="utf-8",
    )

    summary = {"config_path": str(config_path)}
    metadata = _load_plot_metadata(summary)

    assert metadata["feature_names"] == ["S2", "S3", "S4"]
    assert metadata["target_names"] == ["E1", "E2", "E3", "E4"]


def test_plot_shap_history_creates_grouped_figure(tmp_path):
    history_path = tmp_path / "history.json"
    history = {
        "total_loss": [0.3, 0.2, 0.1],
        "main_loss": [0.2, 0.15, 0.09],
        "shap_loss": [0.01, 0.01, 0.02],
        "tikhonov_loss": [0.02, 0.025, 0.03],
        "shap_contribution": [1e-5, 2e-5, 2.5e-5],
        "tikhonov_contribution": [3e-4, 2.8e-4, 2.7e-4],
        "adaptive_gamma": [0.02, 0.05, 0.1],
        "convergence_slowdown": [0.7, 0.5, 0.3],
        "regularization_share": [0.03, 0.05, 0.06],
        "shap_consistency": [0.1, 0.11, 0.12],
        "shap_sparsity": [0.7, 0.69, 0.68],
        "shap_faithfulness": [0.01, 0.012, 0.013],
        "shap_stability": [0.001, 0.0012, 0.0011],
        "shap_weight_consistency": [0.2, 0.22, 0.24],
        "shap_weight_sparsity": [0.3, 0.28, 0.26],
        "shap_weight_faithfulness": [0.25, 0.24, 0.23],
        "shap_weight_stability": [0.25, 0.26, 0.27],
        "shap_scale_factor": [1000, 1200, 1300],
    }
    history_path.write_text(json.dumps(history), encoding="utf-8")

    output = plot_shap_history(tmp_path, {"history": history_path.name}, "unit", tmp_path)

    assert output is not None
    assert output.exists()


def test_plot_regularization_summary_creates_figure(tmp_path):
    summary = {
        "timestamp": "unit",
        "diagnostics": {
            "regularization": {
                "active_components": ["consistency", "sparsity"],
                "dominant_regularizer": "tikhonov",
                "dominant_shap_component": "consistency",
                "shap_contribution": {"mean": 1.0e-4},
                "tikhonov_contribution": {"mean": 4.0e-4},
                "regularization_share": {"mean": 0.06, "last": 0.05, "max": 0.08},
                "component_terms": {
                    "consistency": {"mean": 0.1},
                    "sparsity": {"mean": 0.7},
                },
                "weighted_component_signal": {
                    "consistency": {"mean": 2.0e-4},
                    "sparsity": {"mean": 1.8e-4},
                },
            }
        },
    }

    output = plot_regularization_summary(summary, tmp_path)

    assert output is not None
    assert output.exists()
