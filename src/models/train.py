"""DVC stage: train and compare fraud models, track everything in MLflow.

The most important file in the project. With ``train.model_type: auto`` it
trains **four model families** — XGBoost, LightGBM, random forest and logistic
regression — each as its own (nested) MLflow run, then keeps the best one by
validation **PR-AUC** (``selection_metric``) and registers it. Each candidate
run:

1. logs every hyperparameter and the tuned decision threshold,
2. computes the full fraud-focused metric suite on the validation split
   (reusing :func:`evaluate.compute_metrics` so numbers are stage-consistent),
3. generates and logs explainability + diagnostic plots (SHAP, confusion
   matrix, PR/ROC curves) and feature importances — SHAP artifacts are a
   regulatory expectation in real fintech pipelines,
4. logs a probability-returning pyfunc wrapper (plus the native XGBoost
   model for the XGBoost candidate).

The parent run logs the comparison table/chart; the winner is registered and
promoted to ``Staging`` *only if* it clears the performance gate.

Run as a DVC stage; uses ``print`` for stage output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed in CI/containers

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (  # noqa: E402
    EXPERIMENT_NAME,
    MODEL_COMPARISON_PATH,
    MODEL_FEATURE_COLUMNS,
    REGISTERED_MODEL_NAME,
    SQL_FEATURES_ENABLED,
    TARGET_COLUMN,
    TEST_PATH,
    THRESHOLD_PATH,
    TRAIN_METRICS_PATH,
    TRAIN_OVERRIDES,
    TRAIN_PATH,
    VAL_PATH,
    Config,
    TrainConfig,
    load_config,
)
from src.models.evaluate import RUN_INFO_PATH, compute_metrics
from src.models.predict import (  # noqa: E402
    CALIBRATOR_ARTIFACT_KEY,
    SKLEARN_ARTIFACT_KEY,
    XGB_ARTIFACT_KEY,
    FraudProbaModel,
    apply_calibrator,
    classify,
    fit_calibrator,
    save_calibrator,
    save_pickled_model,
    save_xgb_model,
)

# Any fitted sklearn-compatible classifier (XGBoost, LightGBM, RF, LogReg).
Classifier: TypeAlias = Any

SHAP_SAMPLE_SIZE = 500
THRESHOLD_EPS = 1e-8


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and validation parquet splits."""
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(VAL_PATH)


def apply_train_overrides(train_cfg: TrainConfig, y_train: pd.Series) -> TrainConfig:
    """Apply the active dataset's per-dataset hyperparameter overrides.

    Supports ``scale_pos_weight: auto`` (computed as negatives/positives from the
    training labels). Lets one global ``train`` section serve many datasets while
    each still gets an imbalance weight and recall floor appropriate to its base
    rate.
    """
    overrides = dict(TRAIN_OVERRIDES)
    if overrides.get("scale_pos_weight") == "auto":
        y = np.asarray(y_train)
        pos = max(int((y == 1).sum()), 1)
        overrides["scale_pos_weight"] = round(float((y == 0).sum()) / pos, 2)
    valid = {f for f in train_cfg.__dataclass_fields__}
    overrides = {k: v for k, v in overrides.items() if k in valid}
    return replace(train_cfg, **overrides) if overrides else train_cfg


def resolve_model_type(cfg: TrainConfig, model_type: str | None = None) -> str:
    """Concrete model family for a single build (``auto`` defaults to XGBoost)."""
    chosen = model_type or cfg.model_type
    return "xgboost" if chosen == "auto" else chosen


def candidate_model_types(cfg: TrainConfig) -> list[str]:
    """Model families to train: every candidate for ``auto``, else just one."""
    return list(cfg.candidates) if cfg.model_type == "auto" else [cfg.model_type]


def build_model(cfg: TrainConfig, model_type: str | None = None) -> Classifier:
    """Construct one classifier configured for the imbalanced problem.

    All four families handle the 577:1 imbalance through the *loss* rather than
    resampling: ``scale_pos_weight`` for the boosters, ``class_weight`` for the
    scikit-learn models. Boosters share the tuned depth / learning-rate /
    subsampling hyperparameters so the comparison is like-for-like.

    Args:
        cfg: Training configuration.
        model_type: Override ``cfg.model_type`` (``auto`` -> ``xgboost``).
    """
    kind = resolve_model_type(cfg, model_type)
    if kind == "xgboost":
        # ``aucpr`` eval metric: PR-AUC is the right summary under imbalance.
        return xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            scale_pos_weight=cfg.scale_pos_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_seed,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )
    if kind == "lightgbm":
        # min_child_weight/reg_lambda are essential here: with LightGBM's
        # default min_child_weight=1e-3, scale_pos_weight=24 at 577:1 lets
        # near-empty leaves take extreme values and val PR-AUC collapses to
        # ~0.2-0.3; with a hessian floor of 1 it is ~0.86 (measured).
        return LGBMClassifier(
            min_child_weight=1.0,
            reg_lambda=1.0,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            num_leaves=2**cfg.max_depth,
            learning_rate=cfg.learning_rate,
            scale_pos_weight=cfg.scale_pos_weight,
            subsample=cfg.subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=cfg.random_seed,
            n_jobs=-1,
        )
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=2000,
            random_state=cfg.random_seed,
        )
    raise ValueError(f"Unknown model_type: {kind!r}")


def _cost_optimal_threshold(
    probs: np.ndarray, y: np.ndarray, cost_fn: float, cost_fp: float
) -> float:
    """Threshold minimising expected cost = cost_fn*FN + cost_fp*FP.

    The business framing of fraud's asymmetric errors: a missed fraud usually
    costs far more than a false alarm, so the optimal cut-off is wherever the
    marginal fraud caught stops being worth the false positives it adds.

    Computed in O(n log n) via a single descending sort plus cumulative sums.
    The obvious implementation — loop over every unique score, re-scanning the
    full array each time — is O(n^2) and does not finish on a realistically
    sized validation split (~42k rows here).
    """
    scores = np.asarray(probs, dtype=float)
    labels = np.asarray(y).astype(int)
    if scores.size == 0:
        return 0.5

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Cutting at index i means predicting positive for rows 0..i, so the running
    # sums give TP/FP directly and FN is whatever positives are left behind.
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    cost = cost_fn * (int(sorted_labels.sum()) - tp) + cost_fp * fp

    # Only cut where the score actually changes: rows 0..i are exactly
    # {score >= sorted_scores[i]} when sorted_scores[i] != sorted_scores[i + 1].
    boundary = np.flatnonzero(np.append(np.diff(sorted_scores) != 0, True))
    boundary_costs = cost[boundary]
    # Ties resolve to the SMALLEST qualifying threshold — the largest index in
    # descending order — matching the original ascending scan's behaviour.
    best = boundary[np.flatnonzero(boundary_costs == boundary_costs.min())[-1]]
    return float(sorted_scores[best])


def find_optimal_threshold(
    model: Classifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = 0.85,
    threshold_strategy: str = "recall",
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    probs: np.ndarray | None = None,
) -> float:
    """Choose the decision threshold for the fraud class (Class=1).

    With ``threshold_strategy="recall"`` (default): pick the **highest-precision**
    threshold whose validation recall is at least ``min_recall``. This is the
    right objective at 577:1 imbalance, where a missed fraud costs far more than
    a false alarm; plain-F1 maximisation instead drifts to a precision-heavy
    point that sacrifices recall. If no threshold reaches the recall floor, fall
    back to maximising F1, and to the neutral 0.5 cut-off when the
    precision-recall curve is empty (e.g. a split with no frauds).

    With ``threshold_strategy="cost"``: minimise expected cost
    ``cost_fn*FN + cost_fp*FP`` directly (see :func:`_cost_optimal_threshold`).

    ``probs`` may be supplied directly (e.g. calibrated validation scores) to
    select the threshold in the same space the model will serve in.
    """
    if probs is None:
        probs = model.predict_proba(x_val)[:, 1]
    if int(np.asarray(y_val).sum()) == 0:
        return 0.5
    if threshold_strategy == "cost":
        return _cost_optimal_threshold(probs, np.asarray(y_val), cost_fn, cost_fp)
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    if thresholds.size == 0:
        return 0.5

    # precision/recall have one more element than thresholds; align by dropping
    # the trailing point (recall=0, precision=1) that has no threshold.
    prec, rec = precisions[:-1], recalls[:-1]
    meets_floor = rec >= min_recall
    if meets_floor.any():
        # thresholds are ascending and recall is (monotonically) non-increasing
        # in threshold, so the LARGEST threshold still meeting the recall floor
        # is the boundary operating point with the best attainable precision at
        # that recall. Picking the raw argmax-precision instead can latch onto a
        # noisy near-zero-threshold spike, so we take the boundary for stability.
        candidate_idx = np.where(meets_floor)[0]
        best = int(candidate_idx[-1])
        return float(thresholds[best])

    # Fallback: maximise F1 when the recall floor is unreachable.
    f1_scores = 2 * (prec * rec) / (prec + rec + THRESHOLD_EPS)
    return float(thresholds[int(np.argmax(f1_scores))]) if f1_scores.size else 0.5


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, out: Path) -> None:
    """Save a row-normalised confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format=".3f", colorbar=False)
    ax.set_title("Normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_pr_curve(
    y_true: pd.Series, probs: np.ndarray, threshold: float, out: Path
) -> None:
    """Save the precision-recall curve with the operating point marked."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    op_idx = int(np.argmin(np.abs(thresholds - threshold)))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recalls, precisions, label="PR curve")
    ax.scatter(
        recalls[op_idx],
        precisions[op_idx],
        color="red",
        zorder=5,
        label=f"operating point @ {threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (fraud class)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_roc_curve(y_true: pd.Series, probs: np.ndarray, out: Path) -> None:
    """Save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_shap_summary(model: Classifier, x_val: pd.DataFrame, out: Path) -> None:
    """Save a SHAP beeswarm summary plot on a validation sample."""
    sample = x_val.sample(n=min(SHAP_SAMPLE_SIZE, len(x_val)), random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):  # some versions: one array per class
        shap_values = shap_values[-1]
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def log_plots(
    model: Classifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float,
    artifact_dir: Path,
    probs: np.ndarray,
    with_shap: bool = True,
) -> None:
    """Generate all diagnostic plots into ``artifact_dir`` and log them.

    ``probs`` must be the SAME scores the threshold was selected from and the
    metrics were computed on. Recomputing raw ``predict_proba`` here instead
    would silently mismatch whenever calibration is enabled: the operating point
    is in calibrated space, so plotting it over an uncalibrated curve puts the
    marker — and the confusion matrix — at the wrong place.
    """
    y_pred = classify(probs, threshold)

    cm_path = artifact_dir / "confusion_matrix.png"
    pr_path = artifact_dir / "pr_curve.png"
    roc_path = artifact_dir / "roc_curve.png"
    shap_path = artifact_dir / "shap_summary.png"

    _save_confusion_matrix(y_val, y_pred, cm_path)
    _save_pr_curve(y_val, probs, threshold, pr_path)
    _save_roc_curve(y_val, probs, roc_path)

    artifacts = [cm_path, pr_path, roc_path]
    # SHAP's TreeExplainer is brittle across shap/xgboost version combos; a
    # missing explainability *plot* must never abort a training run, so it is
    # best-effort and logged only if it succeeds.
    # Only for the gradient-boosted candidates: exact TreeExplainer on a deep
    # random forest is slow, and it does not apply to the linear model.
    if with_shap:
        try:
            _save_shap_summary(model, x_val, shap_path)
            artifacts.append(shap_path)
        except Exception as exc:  # noqa: BLE001 - artifact generation is non-critical
            print(f"[train] SHAP summary skipped ({type(exc).__name__}: {exc})")

    for path in artifacts:
        mlflow.log_artifact(str(path))


def feature_importance_frame(model: Classifier, features: list[str]) -> pd.DataFrame:
    """Importance per feature for any candidate family.

    XGBoost: total gain; LightGBM / random forest: ``feature_importances_``;
    logistic regression: absolute coefficient (features are scaled).
    """
    if isinstance(model, xgb.XGBClassifier):
        gain = model.get_booster().get_score(importance_type="gain")
        names, values = list(gain.keys()), list(gain.values())
    elif hasattr(model, "feature_importances_"):
        names, values = features, list(np.asarray(model.feature_importances_))
    elif hasattr(model, "coef_"):
        names, values = features, list(np.abs(np.asarray(model.coef_)[0]))
    else:
        names, values = [], []
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _log_feature_importance(
    model: Classifier, artifact_dir: Path, features: list[str] | None = None
) -> None:
    """Write feature importances to CSV and log them."""
    frame = feature_importance_frame(model, features or list(MODEL_FEATURE_COLUMNS))
    path = artifact_dir / "feature_importance.csv"
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_models(
    model: Classifier, artifact_dir: Path, calibrator_path: Path | None = None
) -> None:
    """Log a probability pyfunc wrapper (plus the native model for XGBoost).

    When a calibrator is provided it is bundled into the pyfunc so serving
    returns calibrated probabilities transparently.
    """
    if isinstance(model, xgb.XGBClassifier):
        mlflow.xgboost.log_model(model, artifact_path="model_native")
        xgb_path = save_xgb_model(model, artifact_dir / "xgb_model.json")
        artifacts = {XGB_ARTIFACT_KEY: str(xgb_path)}
    else:
        pkl_path = save_pickled_model(model, artifact_dir / "model.pkl")
        artifacts = {SKLEARN_ARTIFACT_KEY: str(pkl_path)}
    if calibrator_path is not None:
        artifacts[CALIBRATOR_ARTIFACT_KEY] = str(calibrator_path)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudProbaModel(),
        artifacts=artifacts,
        code_paths=[
            str(Path(__file__).resolve().parents[1])
        ],  # ship src/ for the wrapper
    )


def maybe_promote_to_staging(run_id: str, f1_fraud: float, target: float) -> None:
    """Register and promote the run's model to Staging if it clears the f1 floor.

    ``target`` is the per-dataset Staging-promotion floor (see
    :func:`train`) — a global default that low-signal profiles override so the
    bar matches each dataset's realistic f1, consistent with the per-dataset
    holdout benchmark gate.
    """
    if f1_fraud < target:
        print(f"[Gate] f1_fraud={f1_fraud:.4f} < {target} — not promoting")
        return
    try:
        # register_model creates the registered model if it doesn't exist yet
        # (create_model_version alone errors on a fresh registry).
        mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
        mlflow.tracking.MlflowClient().transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=mv.version, stage="Staging"
        )
        print(f"[Gate] Promoted model v{mv.version} to Staging")
    except MlflowException as exc:
        # File-store backends (e.g. plain CI) don't support the registry.
        print(f"[Gate] Registry unavailable, skipping promotion: {exc}")


def _persist_outputs(run_id: str, threshold: float) -> None:
    """Write DVC-tracked threshold.json and the run-info pointer file."""
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(
        json.dumps({"threshold": threshold}, indent=2), encoding="utf-8"
    )
    RUN_INFO_PATH.write_text(
        json.dumps({"run_id": run_id, "model_uri": f"runs:/{run_id}/model"}, indent=2),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(THRESHOLD_PATH))


def load_test() -> pd.DataFrame | None:
    """Holdout split, if present (reported per candidate, never used to select)."""
    return pd.read_parquet(TEST_PATH) if Path(TEST_PATH).exists() else None


def train_candidate(
    model_type: str,
    train_cfg: TrainConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train + log ONE model family inside the currently active MLflow run.

    Returns:
        ``{"model_type", "run_id", "threshold", "val": metrics,
        "test": metrics | None}``.
    """
    run = mlflow.active_run()
    assert run is not None, "train_candidate must run inside an MLflow run"
    run_id = str(run.info.run_id)
    features = list(x_train.columns)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        model = build_model(train_cfg, model_type)
        model.fit(x_train, y_train)

        # Optional probability calibration fit on validation. Threshold and
        # logged metrics use the *calibrated* scores so they match what serving
        # produces (the calibrator is bundled into the pyfunc below).
        raw_val = model.predict_proba(x_val)[:, 1]
        calibrator = None
        calibrator_path: Path | None = None
        if train_cfg.calibration != "none":
            calibrator = fit_calibrator(
                raw_val, y_val.to_numpy(), train_cfg.calibration
            )
            calibrator_path = save_calibrator(
                calibrator, artifact_dir / "calibrator.pkl"
            )
            probs = apply_calibrator(calibrator, raw_val)
        else:
            probs = raw_val

        threshold = find_optimal_threshold(
            model,
            x_val,
            y_val,
            min_recall=train_cfg.min_recall,
            threshold_strategy=train_cfg.threshold_strategy,
            cost_fn=train_cfg.cost_fn,
            cost_fp=train_cfg.cost_fp,
            probs=probs,
        )
        metrics = compute_metrics(y_val, probs, threshold)

        # Holdout metrics are *reported* for the comparison table only — the
        # winner is chosen on validation, so the test split stays untouched by
        # model selection.
        test_metrics: dict[str, float] | None = None
        if test_df is not None:
            test_probs = model.predict_proba(test_df[features])[:, 1]
            if calibrator is not None:
                test_probs = apply_calibrator(calibrator, test_probs)
            test_metrics = compute_metrics(
                test_df[TARGET_COLUMN], test_probs, threshold
            )

        mlflow.set_tags(
            {
                "model_type": model_type,
                "feature_set": "sql" if SQL_FEATURES_ENABLED else "base",
            }
        )
        mlflow.log_params(train_cfg.__dict__)
        mlflow.log_param("model_family", model_type)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
        if test_metrics:
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"}
            )

        log_plots(
            model,
            x_val,
            y_val,
            threshold,
            artifact_dir,
            probs,
            with_shap=model_type in {"xgboost", "lightgbm"},
        )
        _log_feature_importance(model, artifact_dir, features)
        _log_models(model, artifact_dir, calibrator_path)

    print(
        f"[train] {model_type:<14} val PR-AUC={metrics['avg_precision']:.4f} "
        f"recall={metrics['recall_fraud']:.4f} "
        f"precision={metrics['precision_fraud']:.4f}"
    )
    return {
        "model_type": model_type,
        "run_id": run_id,
        "threshold": threshold,
        "val": metrics,
        "test": test_metrics,
    }


def select_best(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Winner = highest validation ``metric`` (ties -> first candidate)."""
    return max(results, key=lambda r: float(r["val"][metric]))


def _save_comparison_chart(
    results: list[dict[str, Any]], metric: str, out: Path
) -> None:
    """Bar chart of validation PR-AUC / recall / precision per model family."""
    names = [r["model_type"] for r in results]
    keys = [
        (metric, "PR-AUC"),
        ("recall_fraud", "Recall"),
        ("precision_fraud", "Precision"),
    ]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (key, label) in enumerate(keys):
        values = [r["val"][key] for r in results]
        ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (validation)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _comparison_payload(
    results: list[dict[str, Any]], best: dict[str, Any], metric: str
) -> dict[str, Any]:
    """JSON-serialisable comparison summary (DVC metric + README source)."""
    return {
        "feature_set": "sql" if SQL_FEATURES_ENABLED else "base",
        "selection_metric": metric,
        "best_model": best["model_type"],
        "best_run_id": best["run_id"],
        "models": {
            r["model_type"]: {
                "run_id": r["run_id"],
                "threshold": r["threshold"],
                "val": r["val"],
                "test": r["test"],
            }
            for r in results
        },
    }


def train(cfg: Config) -> str:
    """Train every candidate family, keep the best by PR-AUC, register it.

    Returns:
        The MLflow run id of the winning candidate's run.
    """
    train_df, val_df = load_data()
    features = [c for c in MODEL_FEATURE_COLUMNS if c in train_df.columns]
    x_train, y_train = train_df[features], train_df[TARGET_COLUMN]
    x_val, y_val = val_df[features], val_df[TARGET_COLUMN]
    test_df = load_test()

    train_cfg = apply_train_overrides(cfg.train, y_train)
    metric = train_cfg.selection_metric
    model_types = candidate_model_types(train_cfg)

    mlflow.set_experiment(EXPERIMENT_NAME)
    feature_set = "sql" if SQL_FEATURES_ENABLED else "base"
    with mlflow.start_run(run_name=f"comparison-{feature_set}") as parent:
        mlflow.set_tags({"stage": "model-comparison", "feature_set": feature_set})
        results: list[dict[str, Any]] = []
        for model_type in model_types:
            with mlflow.start_run(run_name=model_type, nested=True):
                results.append(
                    train_candidate(
                        model_type, train_cfg, x_train, y_train, x_val, y_val, test_df
                    )
                )

        best = select_best(results, metric)
        comparison = _comparison_payload(results, best, metric)
        mlflow.log_param("best_model", best["model_type"])
        mlflow.log_param("candidates", ",".join(model_types))
        mlflow.log_metric(f"best_{metric}", float(best["val"][metric]))
        with tempfile.TemporaryDirectory() as tmp:
            chart = Path(tmp) / "model_comparison.png"
            _save_comparison_chart(results, metric, chart)
            mlflow.log_artifact(str(chart))
            table = Path(tmp) / "model_comparison.json"
            table.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(table))
        print(
            f"[train] best model = {best['model_type']} "
            f"({metric}={best['val'][metric]:.4f}, parent run {parent.info.run_id})"
        )

    # Outputs point at the WINNER's run (its pyfunc is what gets served).
    run_id = str(best["run_id"])
    threshold = float(best["threshold"])
    metrics = {**best["val"], "model_type": best["model_type"]}
    with mlflow.start_run(run_id=run_id):
        _persist_outputs(run_id, threshold)

    TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # Staging-promotion floor: per-dataset override (train_overrides) or the
    # global default — keeps the bar consistent with each dataset's gate.
    promotion_floor = float(
        TRAIN_OVERRIDES.get(
            "performance_threshold", cfg.monitoring.performance_threshold
        )
    )
    print(f"[train] run_id={run_id} metrics={metrics}")
    maybe_promote_to_staging(run_id, float(best["val"]["f1_fraud"]), promotion_floor)
    return run_id


def main() -> int:
    """Stage entrypoint."""
    train(load_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
