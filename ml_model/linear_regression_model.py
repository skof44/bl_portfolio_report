"""
Ridge-регрессия с One-Hot Encoding для предсказания ошибки прогнозов аналитиков.

Target: Y = |fact_earn_rate_ann - potential_earn_rate_ann|

Обучение: хронологическое разбиение 80/20 + подбор alpha через TimeSeriesSplit
на обучающей выборке (без look-ahead bias).
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_ALPHA = 10.0
ALPHA_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
N_SPLITS_CV = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"
ARTIFACTS_DIR = DATA_DIR / "model_artifacts"

TRAIN_PATH = DATA_DIR / "SIGNALS_DATA.xlsx"
FORECASTS_PATH = DATA_DIR / "analyst_forecasts_2026_with_predictions.csv"

CATEGORICAL_FEATURES = ["analyst_name", "ticker"]
NUMERICAL_FEATURES = [
    "term",
    "potential_earn_rate_ann",
    "risk_free_rate_ann",
]

DROP_COLS = [
    "signal_id",
    "uid",
    "end_dt",
    "close_price",
    "fact_earn_rate_ann",
    "initial_price",
    "target_price",
]

TARGET_COL = "target_y"

ANALYST_MAP = {
    "ЦИFРА брокер": "Цифра Брокер",
    "Сбер Инвестиции": "Сбер",
    "SberCIB": "Сбер",
    "Т-Инвестиций": "Тинькофф Инвестиции",
}


@dataclass
class PreprocessingState:
    """Состояние предобработки, зафиксированное на train."""

    top_tickers: list[str]
    analyst_map: dict[str, str]
    top_n_tickers: int = 30

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> PreprocessingState:
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Неподдерживаемый формат: {path}")


def add_target(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Добавляет target_y и удаляет строки с нулевой ошибкой."""
    out = df.copy()
    if "fact_earn_rate_ann" not in out.columns or "potential_earn_rate_ann" not in out.columns:
        raise KeyError("Для target нужны fact_earn_rate_ann и potential_earn_rate_ann")

    out[TARGET_COL] = (out["fact_earn_rate_ann"] - out["potential_earn_rate_ann"]).abs()
    zero_mask = out[TARGET_COL] == 0
    n_zero = int(zero_mask.sum())
    if n_zero:
        out = out[~zero_mask].copy()
    return out, n_zero


def fit_preprocessing(df: pd.DataFrame, top_n_tickers: int = 30) -> PreprocessingState:
    tickers = df["ticker"].fillna("UNKNOWN").astype(str)
    top_tickers = tickers.value_counts().head(top_n_tickers).index.tolist()
    return PreprocessingState(
        top_tickers=top_tickers,
        analyst_map=ANALYST_MAP.copy(),
        top_n_tickers=top_n_tickers,
    )


def apply_preprocessing(df: pd.DataFrame, state: PreprocessingState) -> pd.DataFrame:
    out = df.copy()

    if "analyst_name" in out.columns:
        out["analyst_name"] = out["analyst_name"].fillna("UNKNOWN").astype(str)
        out["analyst_name"] = out["analyst_name"].replace(state.analyst_map)

    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].fillna("UNKNOWN").astype(str)
        out.loc[~out["ticker"].isin(state.top_tickers), "ticker"] = "rare_ticker"

    drop_cols = [c for c in DROP_COLS if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")
    return out


def load_training_frame(path: Path = TRAIN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = _read_table(path)
    df, _ = add_target(df)
    if "create_dt" not in df.columns:
        raise KeyError("create_dt отсутствует — невозможно хронологическое разбиение")
    df["create_dt"] = pd.to_datetime(df["create_dt"])
    return df.sort_values("create_dt").reset_index(drop=True)


def load_forecasts_frame(path: Path = FORECASTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = _read_table(path)
    if "create_dt" in df.columns:
        df["create_dt"] = pd.to_datetime(df["create_dt"])
    return df


def chronological_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(df) * (1 - test_size))
    if n_train < 1 or n_train >= len(df):
        raise ValueError(f"Некорректный размер train при n={len(df)}")
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def prepare_xy(df: pd.DataFrame, state: PreprocessingState) -> tuple[pd.DataFrame, pd.Series | None]:
    processed = apply_preprocessing(df, state)
    y = processed.pop(TARGET_COL) if TARGET_COL in processed.columns else None
    x = processed.drop(columns=["create_dt"], errors="ignore")
    return x, y


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def build_pipeline(alpha: float = DEFAULT_ALPHA) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
        ]
    )


def select_alpha_time_series_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    alpha_grid: list[float] = ALPHA_GRID,
    n_splits: int = N_SPLITS_CV,
) -> tuple[float, list[dict[str, float]]]:
    """Подбор alpha по expanding-window TimeSeriesSplit на train."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results: list[dict[str, float]] = []

    for alpha in alpha_grid:
        fold_mae: list[float] = []
        for train_idx, val_idx in tscv.split(x_train):
            model = build_pipeline(alpha=alpha)
            model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(x_train.iloc[val_idx])
            fold_mae.append(mean_absolute_error(y_train.iloc[val_idx], pred))
        cv_results.append({"alpha": alpha, "cv_mae": float(np.mean(fold_mae))})

    best = min(cv_results, key=lambda r: r["cv_mae"])
    return float(best["alpha"]), cv_results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        "MAPE_%": mape,
        "MAE_%": mae * 100,
        "RMSE_%": rmse * 100,
        "MedAE_%": medae * 100,
    }


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        warnings.warn(f"matplotlib не установлен: {exc}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true * 100, y_pred * 100, alpha=0.5, edgecolors="none")
    lim = max(y_true.max(), y_pred.max()) * 100
    ax.plot([0, lim], [0, lim], "r--", lw=2)
    ax.set_xlabel("Фактическая ошибка, %")
    ax.set_ylabel("Предсказанная ошибка, %")
    ax.set_title("Ridge + OHE: test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "lr_predicted_vs_actual.png", dpi=200)
    plt.close(fig)

    residuals = (y_true - y_pred) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred * 100, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Предсказанная ошибка, %")
    ax.set_ylabel("Остатки, п.п.")
    ax.set_title("Residual plot (test)")
    fig.tight_layout()
    fig.savefig(figures_dir / "lr_residuals.png", dpi=200)
    plt.close(fig)

    print(f"Графики сохранены в: {figures_dir}")


# ---------------------------------------------------------------------------
# Артефакты
# ---------------------------------------------------------------------------

def save_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    state: PreprocessingState,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "linear_regression_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    state.save(artifacts_dir / "preprocessing_config.json")
    print(f"Артефакты сохранены в: {artifacts_dir}")


def load_pipeline(artifacts_dir: Path = ARTIFACTS_DIR) -> Pipeline:
    with open(artifacts_dir / "linear_regression_pipeline.pkl", "rb") as f:
        return pickle.load(f)


def load_preprocessing_state(artifacts_dir: Path = ARTIFACTS_DIR) -> PreprocessingState:
    return PreprocessingState.load(artifacts_dir / "preprocessing_config.json")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_frame(df: pd.DataFrame, pipeline: Pipeline, state: PreprocessingState) -> np.ndarray:
    x, _ = prepare_xy(df, state)
    y_pred = pipeline.predict(x)
    return np.maximum(y_pred, 0.0)


def export_bl_forecasts(
    forecasts_path: Path = FORECASTS_PATH,
    output_path: Path | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    """Обновляет analyst_forecasts CSV предсказаниями y_pred."""
    pipeline = load_pipeline(artifacts_dir)
    state = load_preprocessing_state(artifacts_dir)
    df = load_forecasts_frame(forecasts_path)

    base_cols = [
        c
        for c in [
            "create_dt",
            "ticker",
            "analyst_name",
            "term",
            "potential_earn_rate_ann",
            "risk_free_rate_ann",
        ]
        if c in df.columns
    ]
    out = df[base_cols].copy()
    out["y_pred"] = predict_frame(df, pipeline, state)

    target = output_path or forecasts_path
    out.to_csv(target, index=False, encoding="utf-8")
    print(f"BL-прогнозы обновлены: {target} ({len(out)} строк)")
    return target


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def main(train_path: Path = TRAIN_PATH) -> dict[str, Any]:
    print("=" * 60)
    print("Ridge + OHE | хронологический train/test + TimeSeriesSplit CV")
    print("=" * 60)

    raw_full = _read_table(train_path)
    raw_full["create_dt"] = pd.to_datetime(raw_full["create_dt"])
    _, n_removed_zero = add_target(raw_full)
    raw = load_training_frame(train_path)
    print(f"Загружено записей (после фильтра target=0): {len(raw)}")
    if n_removed_zero:
        print(f"  Удалено {n_removed_zero} сигналов с target_y = 0")

    train_raw, test_raw = chronological_split(raw, test_size=TEST_SIZE)
    state = fit_preprocessing(train_raw)

    x_train, y_train = prepare_xy(train_raw, state)
    x_test, y_test = prepare_xy(test_raw, state)
    assert y_train is not None and y_test is not None

    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print(
        f"Train период: {train_raw['create_dt'].min().date()} — "
        f"{train_raw['create_dt'].max().date()}"
    )
    print(
        f"Test период:  {test_raw['create_dt'].min().date()} — "
        f"{test_raw['create_dt'].max().date()}"
    )

    best_alpha, cv_results = select_alpha_time_series_cv(x_train, y_train)
    print(f"\nЛучший alpha (TimeSeriesSplit CV на train): {best_alpha}")
    for row in cv_results:
        print(f"  alpha={row['alpha']:>6}: CV MAE = {row['cv_mae']*100:.2f}%")

    pipeline = build_pipeline(alpha=best_alpha)
    pipeline.fit(x_train, y_train)

    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    train_metrics = compute_metrics(y_train.values, y_train_pred)
    test_metrics = compute_metrics(y_test.values, y_test_pred)

    print("\n--- Train metrics ---")
    for k in ("MAE_%", "RMSE_%", "MedAE_%", "R2", "MAPE_%"):
        v = train_metrics[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n--- Test metrics ---")
    for k in ("MAE_%", "RMSE_%", "MedAE_%", "R2", "MAPE_%"):
        v = test_metrics[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    plot_results(y_test.values, y_test_pred)

    x_all, y_all = prepare_xy(raw, state)
    y_all_pred = pipeline.predict(x_all)
    median_y_pred = float(np.median(y_all_pred))

    summary: dict[str, Any] = {
        "model_type": "Ridge_OHE",
        "split": "chronological_80_20",
        "cv": "TimeSeriesSplit",
        "best_alpha": best_alpha,
        "alpha_grid_cv": cv_results,
        "train_period": {
            "start": str(train_raw["create_dt"].min().date()),
            "end": str(train_raw["create_dt"].max().date()),
        },
        "test_period": {
            "start": str(test_raw["create_dt"].min().date()),
            "end": str(test_raw["create_dt"].max().date()),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_removed_zero_target": n_removed_zero,
        "median_y_pred": median_y_pred,
    }
    save_artifacts(pipeline, summary, state)

    if FORECASTS_PATH.exists():
        export_bl_forecasts(FORECASTS_PATH, artifacts_dir=ARTIFACTS_DIR)

    return summary


if __name__ == "__main__":
    main()
