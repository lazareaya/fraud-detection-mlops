"""Tests for the DuckDB SQL feature engineering and its train/serve parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.config import FEATURE_COLUMNS, SCALED_COLUMNS, TARGET_COLUMN, PreprocessConfig
from src.data import preprocess as preprocess_mod
from src.data.features import (
    SQL_FEATURE_COLUMNS,
    add_sql_features,
    fit_hourly_stats,
    load_hourly_stats,
    save_hourly_stats,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            # hours: 0, 0, 1, 25h -> hour 1 (day wrap), 23
            "Time": [10.0, 3_000.0, 3_700.0, 90_000.0, 86_399.0],
            "Amount": [10.0, 30.0, 100.0, 0.0, 50.0],
            "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[50, 40, 30, 20, 10],  # non-default index must survive
    )


def test_sql_features_values_and_order() -> None:
    """hour / log amount / window-mean deviation are correct, order preserved."""
    out = add_sql_features(_frame())
    assert list(out.index) == [50, 40, 30, 20, 10]
    assert list(out["V1"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(out["hour_of_day"]) == [0.0, 0.0, 1.0, 1.0, 23.0]
    np.testing.assert_allclose(out["log_amount"], np.log1p(_frame()["Amount"]))
    # hour 0 mean = 20 ; hour 1 mean = 50 ; hour 23 mean = 50
    np.testing.assert_allclose(
        out["amount_vs_hour_mean"], [-10.0, 10.0, 50.0, -50.0, 0.0]
    )


def test_join_path_matches_window_on_train_and_falls_back() -> None:
    """Joining train stats reproduces the window result; unseen hour -> global."""
    train = _frame()
    stats = fit_hourly_stats(train)
    via_window = add_sql_features(train)
    via_join = add_sql_features(train, stats)
    np.testing.assert_allclose(
        via_window["amount_vs_hour_mean"], via_join["amount_vs_hour_mean"]
    )
    unseen = pd.DataFrame({"Time": [5 * 3600.0], "Amount": [42.0]})  # hour 5
    out = add_sql_features(unseen, stats)
    assert out["amount_vs_hour_mean"].iloc[0] == pytest.approx(42.0 - 38.0)


def test_stats_roundtrip_and_missing_inputs(tmp_path: Path) -> None:
    stats = fit_hourly_stats(_frame())
    path = save_hourly_stats(stats, tmp_path / "s.json")
    assert load_hourly_stats(path) == stats
    with pytest.raises(KeyError):
        add_sql_features(pd.DataFrame({"Amount": [1.0]}))


def test_preprocess_with_sql_features(
    raw_df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enabled -> engineered columns land in every split, scaled, stats saved."""
    monkeypatch.setattr(preprocess_mod, "SQL_FEATURES_ENABLED", True)
    monkeypatch.setattr(
        preprocess_mod, "MODEL_SCALED_COLUMNS", SCALED_COLUMNS + SQL_FEATURE_COLUMNS
    )
    cfg = PreprocessConfig(test_size=0.15, val_size=0.15, random_seed=42)
    paths = preprocess_mod.preprocess(
        raw_df, cfg, out_dir=tmp_path / "p", scaler_path=tmp_path / "scaler.pkl"
    )
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(paths[split])
        assert set(SQL_FEATURE_COLUMNS) <= set(frame.columns)
    stats = load_hourly_stats(tmp_path / "hourly_amount_stats.json")
    assert stats["enabled"] is True and len(stats["hourly_mean"]) == 24  # type: ignore[arg-type]


async def test_api_computes_sql_features(
    raw_df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving builds the same engineered feature vector the model trained on."""
    from api import main

    seen: dict[str, pd.DataFrame] = {}

    class _Recorder:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            seen["frame"] = frame
            return np.full(len(frame), 0.1)

    model_cols = FEATURE_COLUMNS + SQL_FEATURE_COLUMNS
    monkeypatch.setattr(main, "SQL_FEATURES_ENABLED", True)
    monkeypatch.setattr(main, "MODEL_FEATURE_COLUMNS", model_cols)
    monkeypatch.setattr(main, "MODEL_SCALED_COLUMNS", SCALED_COLUMNS)
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "audit.jsonl"))
    stats = fit_hourly_stats(raw_df.drop(columns=[TARGET_COLUMN]))
    main.STATE.update(
        model=_Recorder(),
        threshold=0.5,
        scaler=None,
        hourly_stats=stats,
        info={"name": "m", "stage": "Production", "version": "1"},
    )
    payload = {"Time": 7_200.0, "Amount": 25.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    try:
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://t"
        ) as client:
            resp = await client.post("/predict", json=payload)
        assert resp.status_code == 200
        frame = seen["frame"]
        assert list(frame.columns) == list(model_cols)
        assert frame["hour_of_day"].iloc[0] == 2.0
        assert frame["log_amount"].iloc[0] == pytest.approx(np.log1p(25.0))

        main.STATE["hourly_stats"] = None  # stats missing -> not ready
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://t"
        ) as client:
            resp = await client.post("/predict", json=payload)
        assert resp.status_code == 503
    finally:
        main.STATE.update(
            model=None, threshold=None, scaler=None, hourly_stats=None, info={}
        )
