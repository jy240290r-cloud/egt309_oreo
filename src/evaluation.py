"""Model evaluation on the held-out test set.

The test set is used here **once**. Metrics are reported for class 1
(maintenance required): recall (main), precision, F1, F2, PR-AUC, ROC-AUC and
the confusion matrix. Accuracy is included for reference only, because always
predicting "no maintenance" already scores about 80%.

Every model is evaluated at the default threshold (0.5) and at its tuned
threshold. Feature importance combines permutation importance on the test set
(all models) with native importance where the model has one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw plots to files only (no screen needed, e.g. inside Docker)

import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models import get_model
from trainer import TrainedModel

logger = logging.getLogger(__name__)

#: Fixed colour per model, so a model keeps its colour in every plot.
MODEL_COLOURS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Columns that select_best needs in the results table.
SELECTION_COLUMNS: tuple[str, ...] = ("model", "threshold_type", "train_seconds")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the class-1 metrics for one set of probabilities at one threshold.

    PR-AUC and ROC-AUC use the probabilities directly, so they do not depend on
    the threshold. All other metrics use the 0/1 predictions at ``threshold``.

    Args:
        y_true: True 0/1 labels.
        y_prob: Predicted probabilities of class 1.
        threshold: Probability at or above which a row is predicted as 1.

    Returns:
        Dict with ``recall, precision, f1, f2, pr_auc, roc_auc, accuracy, tn, fp, fn, tp``.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def positive_proba(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of class 1 (maintenance required) for every row.

    Args:
        model: A trained model.
        X: Features to score.

    Returns:
        1-D array of probabilities.
    """
    return model.pipeline.predict_proba(X)[:, 1]


class ModelEvaluator:
    """Evaluate trained models on the test set, compare them and write a report.

    Attributes:
        cfg: Full pipeline configuration.
        results: Metrics table from the last ``evaluate`` call (None before).
        importances: Feature-importance table from the last ``feature_importance`` call.
        figures: Paths of the plots written so far.
        best_model: Name chosen by the last ``select_best`` call (None before).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store the config and initialise empty result holders.

        Args:
            cfg: Full pipeline configuration.
        """
        self.cfg = cfg
        self.results: pd.DataFrame | None = None
        self.importances: pd.DataFrame | None = None
        self.figures: list[Path] = []
        self.best_model: str | None = None

    @property
    def dpi(self) -> int:
        """Resolution of saved plots, from ``evaluation.plot_dpi``."""
        return self.cfg["evaluation"]["plot_dpi"]

    def evaluate(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Score every model on the test set at ``threshold.default`` and at its tuned threshold.

        Logs the metrics of each model.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.

        Returns:
            One row per (model, threshold type) with the ``compute_metrics`` columns
            plus ``model``, ``threshold_type``, ``threshold``, ``cv_score`` and ``train_seconds``.
        """
        default_threshold = float(self.cfg["threshold"]["default"])
        rows = []
        for name, model in trained.items():
            y_prob = positive_proba(model, X_test)
            for threshold_type, threshold in (("default", default_threshold), ("tuned", model.threshold)):
                metrics = compute_metrics(y_test, y_prob, threshold)
                rows.append({
                    "model": name,
                    "threshold_type": threshold_type,
                    "threshold": threshold,
                    **metrics,
                    "cv_score": model.cv_score,
                    "train_seconds": model.train_seconds,
                })
                logger.info(
                    "%-22s %-7s thr=%.2f | recall=%.3f precision=%.3f f1=%.3f f2=%.3f pr_auc=%.3f",
                    name, threshold_type, threshold, metrics["recall"], metrics["precision"],
                    metrics["f1"], metrics["f2"], metrics["pr_auc"],
                )
        self.results = pd.DataFrame(rows)
        return self.results

    def feature_importance(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Compute permutation importance on the test set and native importance per model.

        Uses ``evaluation.permutation_repeats``, ``evaluation.permutation_scoring``
        and the config seed. Native importance comes from each model's
        ``native_importance`` (None for HGB).

        Permutation importance shuffles one input column at a time and measures how
        much the score drops. It works for every model and is not biased towards
        features with many unique values, unlike random forest impurity importance.
        It is measured on the raw input columns (before preprocessing), while native
        importance is per engineered feature.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.

        Returns:
            Long table with columns ``model, feature, kind, importance, std``
            (``kind`` is ``"permutation"`` or ``"native"``).
        """
        eval_cfg = self.cfg["evaluation"]
        frames = []
        for name, model in trained.items():
            logger.info("%s: computing permutation importance ...", name)
            perm = permutation_importance(
                model.pipeline, X_test, y_test,
                n_repeats=eval_cfg["permutation_repeats"],
                scoring=eval_cfg["permutation_scoring"],
                random_state=self.cfg["random_state"],
                n_jobs=self.cfg["tuning"]["n_jobs"],
            )
            frames.append(pd.DataFrame({
                "model": name,
                "feature": X_test.columns,
                "kind": "permutation",
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }))

            native = get_model(name, self.cfg).native_importance(model.pipeline)
            if native is not None:
                frames.append(pd.DataFrame({
                    "model": name,
                    "feature": native.index,
                    "kind": "native",
                    "importance": native.to_numpy(),
                    "std": np.nan,  # native importance has no spread
                }))

        self.importances = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["model", "kind", "importance"], ascending=[True, True, False], ignore_index=True)
        )
        return self.importances

    def plot_confusion_matrices(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series, directory: str | Path
    ) -> Path:
        """Plot one confusion matrix per model at its tuned threshold and save the figure.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.
            directory: Folder to write the PNG to.

        Returns:
            Path of the saved figure.
        """
        fig, axes = plt.subplots(1, len(trained), figsize=(4.6 * len(trained), 4.2), squeeze=False)
        for ax, (name, model) in zip(axes[0], trained.items()):
            y_pred = (positive_proba(model, X_test) >= model.threshold).astype(int)
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, labels=[0, 1], display_labels=["No maint.", "Maint."],
                cmap="Blues", values_format=",", colorbar=False, ax=ax,
            )
            ax.set_title(f"{name}\n(threshold = {model.threshold:.2f})", fontsize=10)
            ax.grid(False)
        fig.suptitle("Confusion matrices on the test set (tuned thresholds)")
        fig.tight_layout()
        return self._save_figure(fig, directory, "confusion_matrices.png")

    def plot_pr_curves(
        self, trained: dict[str, TrainedModel], X_test: pd.DataFrame, y_test: pd.Series, directory: str | Path
    ) -> Path:
        """Plot the precision-recall curve of every model on one figure and save it.

        Marks each model's tuned threshold and shows the positive-class baseline.

        Args:
            trained: Output of ``ModelTrainer.fit_all``.
            X_test: Test features.
            y_test: Test target.
            directory: Folder to write the PNG to.

        Returns:
            Path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for colour, (name, model) in zip(MODEL_COLOURS, trained.items()):
            y_prob = positive_proba(model, X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            ax.plot(recall, precision, color=colour, lw=2, label=f"{name} (PR-AUC {pr_auc:.3f})")

            # Where this model actually operates, at its tuned threshold.
            point = compute_metrics(y_test, y_prob, model.threshold)
            ax.scatter(point["recall"], point["precision"], s=70, color=colour,
                       edgecolor="white", linewidth=2, zorder=3)

        baseline = float(np.mean(y_test))
        ax.axhline(baseline, color="#777", ls="--", lw=1, label=f"baseline (positive rate {baseline:.2f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
               title="Precision-recall curves on the test set\n(dots = tuned thresholds)")
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        return self._save_figure(fig, directory, "pr_curves.png")

    def select_best(self, results: pd.DataFrame) -> str:
        """Pick the best model from the tuned-threshold rows of ``results``.

        Ranks by ``evaluation.selection_metric`` (F2). Models within
        ``evaluation.tie_tolerance`` of the best are treated as tied and ranked by
        ``evaluation.tie_breaker`` (PR-AUC), then by shorter training time. A tie
        is logged honestly instead of claiming a clear winner.

        Args:
            results: Output of ``evaluate``.

        Returns:
            Name of the selected model.

        Raises:
            ValueError: If ``results`` is empty or lacks the selection columns.
        """
        eval_cfg = self.cfg["evaluation"]
        metric, tie_breaker = eval_cfg["selection_metric"], eval_cfg["tie_breaker"]
        needed = [*SELECTION_COLUMNS, metric, tie_breaker]
        missing = [col for col in needed if col not in results.columns]
        if results.empty or missing:
            raise ValueError(f"Cannot select a model: results are empty or missing columns {missing}.")

        tuned = results[results["threshold_type"] == "tuned"]
        best_score = tuned[metric].max()
        tied = tuned[tuned[metric] >= best_score - eval_cfg["tie_tolerance"]]
        ranked = tied.sort_values([tie_breaker, "train_seconds"], ascending=[False, True])
        best = str(ranked.iloc[0]["model"])

        if len(tied) > 1:
            logger.info(
                "Models within %.3f %s of each other (tie): %s. Chose %s by %s, then training time.",
                eval_cfg["tie_tolerance"], metric, ", ".join(tied["model"]), best, tie_breaker,
            )
        logger.info("Selected model: %s (%s = %.4f)", best, metric, ranked.iloc[0][metric])
        self.best_model = best
        return best

    def save_report(self, directory: str | Path) -> list[Path]:
        """Write the metrics and importance tables (CSV) and a short text summary.

        Args:
            directory: Target folder (normally ``paths.report_dir``); created if needed.

        Returns:
            Paths of the files written.

        Raises:
            RuntimeError: If ``evaluate`` has not been run yet.
        """
        if self.results is None:
            raise RuntimeError("No results to save: run evaluate() first.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = []

        metrics_path = directory / "metrics.csv"
        self.results.round(4).to_csv(metrics_path, index=False)
        written.append(metrics_path)

        if self.importances is not None:
            importance_path = directory / "feature_importance.csv"
            self.importances.round(4).to_csv(importance_path, index=False)
            written.append(importance_path)

        summary_path = directory / "summary.txt"
        summary_path.write_text(self._summary_text(), encoding="utf-8")
        written.append(summary_path)

        for path in written:
            logger.info("Wrote %s", path)
        return written

    # ------------------------------------------------------------------ helpers

    def _save_figure(self, fig: plt.Figure, directory: str | Path, filename: str) -> Path:
        """Save a figure into ``directory``, close it and remember its path.

        Args:
            fig: Figure to save.
            directory: Target folder; created if needed.
            filename: File name of the PNG.

        Returns:
            Path of the saved figure.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)  # free memory; figures are not shown on screen
        self.figures.append(path)
        logger.info("Saved plot %s", path)
        return path

    def _summary_text(self) -> str:
        """Build the plain-text summary written by ``save_report``.

        Returns:
            Multi-line summary of settings, tuned-threshold results and the selected model.
        """
        columns = ["model", "threshold", "recall", "precision", "f1", "f2", "pr_auc", "roc_auc", "train_seconds"]
        tuned = self.results[self.results["threshold_type"] == "tuned"][columns]
        lines = [
            "EGT309 Predictive Maintenance - evaluation summary",
            "=" * 52,
            f"Feature set      : {self.cfg['features']['feature_set']}",
            f"Tuning           : {'on' if self.cfg['tuning']['enabled'] else 'off'} "
            f"(scoring = {self.cfg['tuning']['scoring']})",
            f"Selection metric : {self.cfg['evaluation']['selection_metric']} at the tuned threshold",
            f"Selected model   : {self.best_model or '(select_best not run)'}",
            "",
            "Test-set results at each model's tuned threshold (class 1 = maintenance required):",
            tuned.round(3).to_string(index=False),
        ]
        if self.figures:
            lines += ["", "Plots:", *[f"  {path.name}" for path in self.figures]]
        return "\n".join(lines) + "\n"