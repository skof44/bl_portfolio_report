"""
Ядро модели Блэка-Литтермана.

Математика по:
  He G., Litterman R. (1999). The Intuition Behind Black-Litterman
  Model Portfolios. Goldman Sachs Investment Management Research.

Обозначения
-----------
π   — вектор равновесных (prior) ожидаемых доходностей, (n,)
δ   — коэффициент неприятия риска рынка
τ   — масштаб неопределённости prior
Σ   — годовая ковариационная матрица доходностей, (n, n)
w   — вектор рыночных (капитализационных) весов, (n,)
P   — матрица прогнозов (picking matrix), (k, n)
Q   — вектор ожидаемых доходностей по прогнозам (годовые), (k,)
Ω   — матрица неопределённости прогнозов (диагональная), (k, k)
μ̂  — апостериорные ожидаемые доходности (posterior), (n,)
Σ̂  — апостериорная ковариационная матрица, (n, n)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class BLViews:
    """Инвесторские прогнозы в формате модели Блэка-Литтермана."""

    P: np.ndarray      # (k, n) матрица прогнозов
    Q: np.ndarray      # (k,)   ожидаемые доходности по прогнозам (годовые)
    Omega: np.ndarray  # (k, k) матрица неопределённости прогнозов
    names: tuple = ()  # текстовые описания прогнозов (опционально)

    @property
    def k(self) -> int:
        return self.P.shape[0]

    @property
    def n(self) -> int:
        return self.P.shape[1]


@dataclass(frozen=True, slots=True)
class BLResult:
    """Результаты применения модели Блэка-Литтермана."""

    tickers: tuple         # имена активов
    pi: np.ndarray         # (n,) равновесные доходности (prior)
    mu_bl: np.ndarray      # (n,) апостериорные ожидаемые доходности
    cov_bl: np.ndarray     # (n, n) апостериорная ковариационная матрица
    weights_mkt: np.ndarray  # (n,) рыночные веса (для сравнения)
    weights_bl: np.ndarray   # (n,) оптимальные веса BL-портфеля
    delta: float
    tau: float
    views: BLViews | None = None  # использованные views (для отчётности)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def infer_risk_aversion(
    mu_market: float,
    rf: float,
    sigma_market: float,
) -> float:
    """
    Оценивает коэффициент неприятия риска δ из наблюдаемых данных.

        δ = (E[Rm] − Rf) / σ²_m

    Типичное значение для глобального рынка: δ ≈ 2.5.
    Для MOEX в 2023-2026: δ может быть выше из-за высокой ключевой ставки.
    """
    variance = sigma_market ** 2
    if variance <= 0:
        raise ValueError("Market variance must be positive.")
    return (mu_market - rf) / variance


def reverse_optimize(
    cov: np.ndarray,
    weights: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Обратная оптимизация: восстанавливает равновесные доходности π.

        π = δ · Σ · w

    Вывод: из задачи максимизации CARA-полезности
        max_w  w'μ − (δ/2) w'Σw
    условие первого порядка при w = w_mkt даёт π = δΣw_mkt.
    """
    return delta * cov @ weights


def build_omega_proportional(
    P: np.ndarray,
    cov: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Матрица неопределённости прогнозов по методу He & Litterman (1999).

        Ω = diag(τ · P · Σ · P')

    Интерпретация: неопределённость каждого прогноза пропорциональна
    дисперсии соответствующего лонг-шорт портфеля.
    """
    raw = tau * P @ cov @ P.T
    return np.diag(np.diag(raw))


def build_omega_idzorek(
    P: np.ndarray,
    cov: np.ndarray,
    tau: float,
    confidences: Sequence[float],
) -> np.ndarray:
    """
    Матрица Ω по методу Idzorek (2005) через уровни уверенности.

        Ω_ii = ((1 − c_i) / c_i) · τ · (P_i Σ P_i')

    confidence = 1.0  →  абсолютная уверенность (Ω_ii → 0)
    confidence → 0.0  →  нет уверенности (Ω_ii → ∞)

    Метод позволяет задавать неопределённость интуитивно: "я уверен
    в этом прогнозе на 60%".
    """
    k = P.shape[0]
    if len(confidences) != k:
        raise ValueError(f"Need {k} confidence values, got {len(confidences)}.")
    tau_psp = np.diag(tau * P @ cov @ P.T)
    omega_diag = np.empty(k)
    for i, conf in enumerate(confidences):
        if not (0.0 < conf < 1.0):
            raise ValueError(f"confidences[{i}] must be in (0, 1), got {conf}.")
        omega_diag[i] = (1.0 - conf) / conf * tau_psp[i]
    return np.diag(omega_diag)


def master_formula(
    pi: np.ndarray,
    cov: np.ndarray,
    views: BLViews,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Мастер-формула Блэка-Литтермана (He & Litterman 1999).

    Основная (long-form) форма:
        M   = (τΣ)⁻¹ + P'Ω⁻¹P
        μ̂  = M⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
        Σ̂  = M⁻¹ + Σ

    Эквивалентная (short-form) форма для кросс-проверки:
        μ̂  = π + τΣP'[τPΣP' + Ω]⁻¹(Q − Pπ)

    Возвращает (mu_bl, cov_bl).
    """
    P, Q = views.P, views.Q
    tau_cov = tau * cov
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(views.Omega)

    # Long-form
    M = tau_cov_inv + P.T @ omega_inv @ P
    M_inv = np.linalg.inv(M)
    mu_bl = M_inv @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
    cov_bl = M_inv + cov

    # Short-form (кросс-проверка эквивалентности)
    A = tau_cov @ P.T @ np.linalg.inv(tau * P @ cov @ P.T + views.Omega)
    mu_alt = pi + A @ (Q - P @ pi)
    if not np.allclose(mu_bl, mu_alt, atol=1e-8):
        logger.warning(
            "master_formula: long-form и short-form расходятся на %.2e",
            np.max(np.abs(mu_bl - mu_alt)),
        )
    else:
        logger.debug("master_formula: long-form ≡ short-form ✓")

    return mu_bl, cov_bl


def compute_bl_weights(
    mu_bl: np.ndarray,
    cov: np.ndarray,
    delta: float,
    *,
    normalize: bool = True,
    long_only: bool = True,
) -> np.ndarray:
    """
    Оптимальные веса BL-портфеля.

        w* = (δΣ)⁻¹ · μ̂

    Аналитическое решение задачи Марковица без ограничений.
    При long_only=True отрицательные веса обнуляются (long-only портфель).
    При normalize=True нормировка к sum(w*) = 1.
    """
    weights = np.linalg.inv(delta * cov) @ mu_bl
    if long_only:
        weights = np.maximum(weights, 0.0)
    if normalize:
        s = weights.sum()
        if abs(s) > 1e-12:
            weights = weights / s
    return weights


# ─── Главная точка входа ──────────────────────────────────────────────────────

def run_black_litterman(
    tickers: Sequence[str],
    cov: np.ndarray,
    weights_mkt: np.ndarray,
    views: BLViews,
    *,
    delta: float = 2.5,
    tau: float = 0.025,
    long_only: bool = True,
) -> BLResult:
    """
    Полный цикл модели Блэка-Литтермана.

        1. Обратная оптимизация : π = δΣw
        2. Мастер-формула       : μ̂, Σ̂
        3. Оптимальные веса     : w* = (δΣ̂)⁻¹μ̂  (нормированы к 1)

    Параметры
    ---------
    tickers     : имена активов (n штук)
    cov         : годовая ковариационная матрица (n × n)
    weights_mkt : рыночные веса (n,), должны суммироваться в 1
    views       : инвесторские прогнозы (BLViews)
    delta       : коэффициент неприятия риска (по умолчанию 2.5)
    tau         : масштаб неопределённости prior (по умолчанию 0.025)
    long_only   : запрет коротких позиций в оптимальном портфеле
    """
    tickers_t = tuple(tickers)
    n = len(tickers_t)

    # Валидация размерностей
    if cov.shape != (n, n):
        raise ValueError(f"cov должна быть ({n}, {n}), получена {cov.shape}.")
    if weights_mkt.shape != (n,):
        raise ValueError(f"weights_mkt должны быть ({n},), получены {weights_mkt.shape}.")
    if views.n != n:
        raise ValueError(f"P должна иметь {n} столбцов, имеет {views.n}.")
    if views.Q.shape[0] != views.k:
        raise ValueError("Длина Q должна совпадать с числом строк P.")

    pi = reverse_optimize(cov, weights_mkt, delta)
    logger.info("π: %s", dict(zip(tickers_t, np.round(pi, 4))))

    mu_bl, cov_bl = master_formula(pi, cov, views, tau)
    logger.info("μ̂: %s", dict(zip(tickers_t, np.round(mu_bl, 4))))

    weights_bl = compute_bl_weights(mu_bl, cov_bl, delta, normalize=True, long_only=long_only)
    logger.info("w*: %s", dict(zip(tickers_t, np.round(weights_bl, 4))))

    return BLResult(
        tickers=tickers_t,
        pi=pi,
        mu_bl=mu_bl,
        cov_bl=cov_bl,
        weights_mkt=weights_mkt,
        weights_bl=weights_bl,
        delta=delta,
        tau=tau,
        views=views,
    )


# ─── MOEX: захардкоженные прогнозы для дипломной работы ──────────────────────

#: Метаданные прогнозов (период 2023–2026, MOEX)
MOEX_VIEWS_META: list[dict] = [
    {
        "name": "SBER: абсолютная доходность 22% годовых",
        "description": (
            "Сбербанк сохраняет рекордную прибыль на фоне высокой ключевой ставки ЦБ РФ; "
            "дивидендная доходность + рост бизнеса оцениваются в 22% годовых."
        ),
        "type": "absolute",
        "long": ["SBER"],
        "short": [],
        "return": 0.22,
        "confidence": 0.60,
    },
    {
        "name": "PLZL: абсолютная доходность 18% годовых",
        "description": (
            "Полюс выигрывает от роста мировых цен на золото и ослабления рубля. "
            "Ожидаемая суммарная доходность с учётом курсового эффекта: 18% годовых."
        ),
        "type": "absolute",
        "long": ["PLZL"],
        "short": [],
        "return": 0.18,
        "confidence": 0.50,
    },
]


def build_moex_views(
    tickers: list[str],
    cov: np.ndarray,
    tau: float = 0.05,
) -> BLViews:
    """
    Строит BLViews для MOEX-портфеля на основе захардкоженных прогнозов.

    Матрица Ω — по методу Idzorek (2005) через уровни уверенности.
    Прогнозы, тикеры которых отсутствуют в портфеле, пропускаются.
    """
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    rows_P: list[np.ndarray] = []
    rows_Q: list[float] = []
    confidences: list[float] = []
    names: list[str] = []

    for meta in MOEX_VIEWS_META:
        longs  = [t for t in meta["long"]  if t in idx]
        shorts = [t for t in meta["short"] if t in idx]

        # Прогноз включается только если все активы присутствуют в портфеле
        if not longs:
            logger.warning("Пропускаю прогноз «%s»: лонг-тикеры не найдены.", meta["name"])
            continue
        if meta["type"] == "relative" and not shorts:
            logger.warning("Пропускаю прогноз «%s»: шорт-тикеры не найдены.", meta["name"])
            continue

        p = np.zeros(n)
        for t in longs:
            p[idx[t]] = 1.0 / len(longs)
        for t in shorts:
            p[idx[t]] = -1.0 / len(shorts)

        rows_P.append(p)
        rows_Q.append(float(meta["return"]))
        confidences.append(float(meta["confidence"]))
        names.append(meta["name"])

    if not rows_P:
        raise RuntimeError("Ни один прогноз не применим к данному набору тикеров.")

    P = np.vstack(rows_P)
    Q = np.array(rows_Q)
    Omega = build_omega_idzorek(P, cov, tau, confidences)

    return BLViews(P=P, Q=Q, Omega=Omega, names=tuple(names))


def build_ml_views(
    tickers: list[str],
    cov: np.ndarray,
    tau: float,
    df_signals: "pd.DataFrame",
    model_mse: float,
) -> BLViews:
    """
    Строит BLViews из предсказаний ML-модели.

    Для каждого тикера, присутствующего и в портфеле, и в df_signals,
    создаётся абсолютный view:
        P_i = [0, ..., 1, ..., 0]  (1 на позиции тикера)
        Q_i = среднее predicted_return по сигналам этого тикера
        Ω_ii = max(model_mse, 1e-8)  (дисперсия ошибки модели)

    Parameters
    ----------
    tickers      : имена активов в портфеле (n штук)
    cov          : годовая ковариационная матрица (n × n)
    tau          : масштаб неопределённости prior
    df_signals   : DataFrame с колонками ['ticker', 'predicted_return']
                   (результат predict_annual_returns из ml_model.py)
    model_mse    : MSE модели на валидации/тесте — дисперсия ошибки предсказания

    Returns
    -------
    BLViews с views для всех тикеров, по которым есть предсказания.
    """
    import pandas as pd

    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    rows_P: list[np.ndarray] = []
    rows_Q: list[float] = []
    names: list[str] = []

    # Усредняем предсказания по тикеру (если сигналов несколько)
    grouped = df_signals.groupby("ticker")["predicted_return"].mean()

    for ticker, pred_return in grouped.items():
        if ticker not in idx:
            logger.warning("Тикер %s из сигналов отсутствует в портфеле — пропускаю.", ticker)
            continue

        p = np.zeros(n)
        p[idx[ticker]] = 1.0

        rows_P.append(p)
        rows_Q.append(float(pred_return))
        names.append(f"{ticker}: ML-прогноз {pred_return:.1%} годовых")

    if not rows_P:
        raise RuntimeError(
            "Ни один тикер из сигналов не найден в портфеле. "
            "Проверьте соответствие тикеров в prices/signals."
        )

    P = np.vstack(rows_P)
    Q = np.array(rows_Q)
    # Ω_ii = дисперсия ошибки модели (MSE).
    # Добавляем epsilon чтобы избежать вырождения при идеальных предсказаниях.
    omega_diag = np.full(P.shape[0], max(float(model_mse), 1e-8))
    Omega = np.diag(omega_diag)

    return BLViews(P=P, Q=Q, Omega=Omega, names=tuple(names))


def build_ridge_views(
    tickers: list[str],
    cov: np.ndarray,
    tau: float,
    df_signals: "pd.DataFrame",
    median_error: float | None = None,
    *,
    rf: float | None = None,
) -> BLViews:
    """
    Строит BLViews из прогнозов Ridge-модели с корректным расчётом Omega.

    Каждая строка df_signals становится отдельным view — это позволяет
    дифференцировать omega для разных аналитиков по одному тикеру.

    Логика Omega (5 шагов):
        1. baseline = tau * diag(P @ Sigma @ P.T)        # K × 1
        2. median_error — медианная ошибка по истории    # скаляр
        3. rel_err = predicted_errors / median_error     # K × 1
        4. alpha = 1.0, beta = 1.0
        5. omega = baseline * (alpha + beta * rel_err^2) # K × 1

    Parameters
    ----------
    tickers      : имена активов в портфеле (n штук)
    cov          : годовая ковариационная матрица (n × n)
    tau          : масштаб неопределённости prior
    df_signals   : DataFrame с колонками ['ticker', 'potential_earn_rate_ann',
                   'risk_free_rate_ann', 'y_pred', 'analyst_name']
    median_error : медианная предсказанная ошибка по истории (если None — загрузит из артефактов)
    rf           : единая безрисковая ставка для Q = total − Rf (как в π и Sharpe).
                   Если None — используется risk_free_rate_ann из каждой строки.

    Returns
    -------
    BLViews с views для каждой строки сигналов (K = число строк).
    """
    import json
    from pathlib import Path

    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    rows_P: list[np.ndarray] = []
    rows_Q: list[float] = []
    rows_pred_err: list[float] = []
    names: list[str] = []

    # Каждая строка — отдельный view (п. 36 чек-листа)
    for _, row in df_signals.iterrows():
        ticker = str(row.get("ticker", ""))
        if ticker not in idx:
            logger.warning("Тикер %s из сигналов отсутствует в портфеле — пропускаю.", ticker)
            continue

        p = np.zeros(n)
        p[idx[ticker]] = 1.0

        total_return = float(row["potential_earn_rate_ann"])
        rf_row = float(rf if rf is not None else row.get("risk_free_rate_ann", 0.0))
        q = total_return - rf_row  # excess return — в одной шкале с π
        pred_err = float(row["y_pred"])
        analyst = str(row.get("analyst_name", "UNKNOWN"))

        rows_P.append(p)
        rows_Q.append(q)
        rows_pred_err.append(pred_err)
        names.append(
            f"{ticker} ({analyst}): excess {q:.1%} "
            f"(total {total_return:.1%}, σ̂_err {pred_err:.1%})"
        )

    if not rows_P:
        raise RuntimeError(
            "Ни один тикер из сигналов не найден в портфеле. "
            "Проверьте соответствие тикеров в prices/signals."
        )

    P = np.vstack(rows_P)
    Q = np.array(rows_Q)
    predicted_errors = np.array(rows_pred_err)

    # --- Расчёт Omega по инструкции ---
    # Шаг 1: Baseline
    baseline = tau * np.diag(P @ cov @ P.T)  # K × 1

    # Шаг 2: Медианная ошибка (загружаем из артефактов, если не передана)
    if median_error is None:
        _metrics_path = Path(__file__).resolve().parent / "data" / "model_artifacts" / "metrics.json"
        if _metrics_path.exists():
            with open(_metrics_path, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
            median_error = _metrics.get("median_y_pred", np.median(predicted_errors))
        else:
            median_error = np.median(predicted_errors)

    # Шаг 3: Относительная ошибка
    rel_err = predicted_errors / max(median_error, 1e-8)  # K × 1

    # Шаг 4: Параметры
    alpha = 1.0
    beta = 1.0
    alpha_q = 10.0  # штраф за агрессивные прогнозы: чем выше Q, тем больше omega

    # Шаг 5: Итоговая omega
    omega = baseline * (alpha + beta * rel_err ** 2) * (1.0 + alpha_q * Q ** 2)  # K × 1
    Omega = np.diag(omega)

    # --- Валидация ---
    ratio = omega / np.maximum(baseline, 1e-12)
    if ratio.max() > 100:
        logger.warning(
            "Omega ratio max = %.1f > 100. Проверь единицы predicted_errors "
            "(должны быть в долях, а не в %%).", ratio.max()
        )

    return BLViews(P=P, Q=Q, Omega=Omega, names=tuple(names))
