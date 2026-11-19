"""SQL feature engineering with DuckDB.

Three behavioural features are derived from the raw ``Time`` / ``Amount``
columns with plain SQL (DuckDB runs it in-process, directly on the pandas
frame — no database server needed):

* ``hour_of_day``         — ``(Time % 86400) / 3600``: fraud is more frequent at
  night, and ``Time`` itself (seconds since the first transaction) is not
  periodic.
* ``log_amount``          — ``LN(1 + Amount)``: tames the heavy right tail of
  transaction amounts (``1 +`` because zero-amount transactions exist).
* ``amount_vs_hour_mean`` — ``Amount - AVG(Amount) OVER (PARTITION BY hour)``:
  how unusual the amount is *for that time of day* (window function).

Leakage / train-serve parity: on the **training** split the hourly mean comes
from the window function; the per-hour means are then persisted
(:func:`fit_hourly_stats`) and **joined** onto validation, test and live API
traffic, so those rows never contribute to the statistic they are compared to.
The same functions run in the preprocessing stage and in the API.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SQL_FEATURE_COLUMNS: tuple[str, ...] = (
    "hour_of_day",
    "log_amount",
    "amount_vs_hour_mean",
)
SQL_FEATURE_INPUTS: tuple[str, ...] = ("Time", "Amount")

_ROW_ID = "__row_id"

# Shared projection: hour bucket + log amount (all columns of the input kept).
_BASE_CTE = """
WITH base AS (
    SELECT
        *,
        CAST(FLOOR(("Time" % 86400) / 3600) AS DOUBLE) AS hour_of_day,
        LN(1 + "Amount")                               AS log_amount
    FROM frame
)
"""

# Training split: hourly mean via a window function over the split itself.
_TRAIN_SQL = (
    _BASE_CTE
    + f"""
SELECT
    * EXCLUDE ({_ROW_ID}),
    "Amount" - AVG("Amount") OVER (PARTITION BY hour_of_day) AS amount_vs_hour_mean
FROM base
ORDER BY {_ROW_ID}
"""
)

# Val / test / serving: join the hourly means learned on the training split.
_APPLY_SQL = (
    _BASE_CTE
    + f"""
SELECT
    b.* EXCLUDE ({_ROW_ID}),
    b."Amount" - COALESCE(s.mean_amount, $global_mean) AS amount_vs_hour_mean
FROM base AS b
LEFT JOIN hourly_stats AS s ON b.hour_of_day = s.hour_of_day
ORDER BY b.{_ROW_ID}
"""
)

_STATS_SQL = (
    _BASE_CTE
    + """
SELECT hour_of_day, AVG("Amount") AS mean_amount, COUNT(*) AS n
FROM base
GROUP BY hour_of_day
ORDER BY hour_of_day
"""
)


def _with_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` with a positional id so SQL output keeps the input row order."""
    return df.reset_index(drop=True).assign(**{_ROW_ID: np.arange(len(df))})


def fit_hourly_stats(train_df: pd.DataFrame) -> dict[str, object]:
    """Per-hour mean ``Amount`` on the training split (``GROUP BY`` in SQL).

    Returns a JSON-serialisable dict: ``{"global_mean": float,
    "hourly_mean": {"0": float, ..., "23": float}}``.
    """
    con = duckdb.connect()
    try:
        con.register("frame", _with_row_id(train_df[list(SQL_FEATURE_INPUTS)]))
        stats = con.execute(_STATS_SQL).df()
    finally:
        con.close()
    return {
        "global_mean": float(train_df["Amount"].mean()),
        "hourly_mean": {
            str(int(h)): float(m)
            for h, m in zip(stats["hour_of_day"], stats["mean_amount"], strict=True)
        },
    }


def add_sql_features(
    df: pd.DataFrame, hourly_stats: dict[str, object] | None = None
) -> pd.DataFrame:
    """Return ``df`` with the SQL features appended (row order + index preserved).

    Args:
        df: Frame containing at least the raw (unscaled) ``Time`` and ``Amount``.
        hourly_stats: ``None`` for the training split (window function over
            ``df`` itself); otherwise the output of :func:`fit_hourly_stats`,
            joined in (val / test / serving).
    """
    missing = [c for c in SQL_FEATURE_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"SQL features need raw columns {missing}")
    con = duckdb.connect()
    try:
        con.register("frame", _with_row_id(df))
        if hourly_stats is None:
            out = con.execute(_TRAIN_SQL).df()
        else:
            hourly = hourly_stats["hourly_mean"]
            assert isinstance(hourly, dict)
            table = pd.DataFrame(
                {
                    "hour_of_day": [float(h) for h in hourly],
                    "mean_amount": [float(v) for v in hourly.values()],
                }
            )
            con.register("hourly_stats", table)
            global_mean = float(hourly_stats["global_mean"])  # type: ignore[arg-type]
            out = con.execute(_APPLY_SQL, {"global_mean": global_mean}).df()
    finally:
        con.close()
    out.index = df.index
    return out


def save_hourly_stats(stats: dict[str, object], path: Path) -> Path:
    """Persist the training-split hourly statistics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_hourly_stats(path: Path) -> dict[str, object]:
    """Load hourly statistics written by :func:`save_hourly_stats`."""
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
