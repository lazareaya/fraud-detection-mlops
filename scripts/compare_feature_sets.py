"""Compare the 4 model families with and without the DuckDB SQL features.

Runs the real ``preprocess`` + ``train`` stages twice — ``MLOPS_SQL_FEATURES=0``
(raw 30 features) then ``=1`` (+ hour_of_day, log_amount,
amount_vs_hour_mean) — so both comparisons land in MLflow as tagged runs, then
writes the 8-row results table used in the README:

* ``metrics/<dataset>/feature_set_comparison.json``
* ``docs/model_comparison.md``
* ``docs/images/model_comparison.png``

Usage (MLflow server running, e.g. ``docker compose up -d mlflow``)::

    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/compare_feature_sets.py

Afterwards run ``dvc repro`` to restore the canonical (params.yaml) outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.getenv("MLOPS_DATASET", "creditcard")
METRICS_DIR = ROOT / "metrics" / DATASET
FEATURE_SETS = {"base": "0", "sql": "1"}
LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "logreg": "Logistic regression",
}


def _run(feature_set: str) -> dict:
    env = {**os.environ, "MLOPS_SQL_FEATURES": FEATURE_SETS[feature_set]}
    for module in ("src.data.preprocess", "src.models.train"):
        print(f"[compare] {feature_set}: python -m {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env, check=True)
    return json.loads((METRICS_DIR / "model_comparison.json").read_text())


def _markdown(results: dict[str, dict]) -> str:
    lines = [
        "| Model | Features | Val PR-AUC | Test PR-AUC | Test ROC-AUC "
        "| Test recall | Test precision | Test F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in LABELS:
        for fs, comp in results.items():
            if model not in comp["models"]:
                continue
            val = comp["models"][model]["val"]
            test = comp["models"][model]["test"]
            star = " **(best)**" if comp["best_model"] == model else ""
            lines.append(
                f"| {LABELS[model]}{star} | {fs} | {val['avg_precision']:.3f} "
                f"| {test['avg_precision']:.3f} | {test['roc_auc']:.3f} "
                f"| {test['recall_fraud']:.3f} | {test['precision_fraud']:.3f} "
                f"| {test['f1_fraud']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _chart(results: dict[str, dict], out: Path) -> None:
    models = [m for m in LABELS if all(m in c["models"] for c in results.values())]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, (fs, comp) in enumerate(results.items()):
        vals = [comp["models"][m]["test"]["avg_precision"] for m in models]
        xs = [j + (i - 0.5) * width for j in range(len(models))]
        bars = ax.bar(xs, vals, width, label=f"{fs} features")
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(range(len(models)), [LABELS[m] for m in models])
    ax.set_ylabel("Test PR-AUC (average precision)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison — with vs without DuckDB SQL features")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    results = {fs: _run(fs) for fs in FEATURE_SETS}
    (METRICS_DIR / "feature_set_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    table = _markdown(results)
    doc = ROOT / "docs" / "model_comparison.md"
    doc.write_text(
        "# Model comparison\n\n"
        "4 model families x 2 feature sets on the Kaggle credit-card fraud data "
        "(stratified 70/15/15 split, deduplicated). Winner selected on "
        "**validation PR-AUC**; test columns are the untouched holdout at each "
        "model's own tuned threshold (highest precision with recall >= floor).\n\n"
        + table
        + "\n![comparison](images/model_comparison.png)\n",
        encoding="utf-8",
    )
    _chart(results, ROOT / "docs" / "images" / "model_comparison.png")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
