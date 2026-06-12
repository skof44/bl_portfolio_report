#!/usr/bin/env python3
"""
Пересчёт таблиц бенчмаркинга для ВКР (ex-ante метрики).

Запуск из корня проекта:
    python scripts/compute_benchmarks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from black_litterman import (  # noqa: E402
    BLViews,
    build_omega_proportional,
    build_ridge_views,
    reverse_optimize,
    run_black_litterman,
)

DATA = ROOT / "data"
DELTA = 2.5
TAU = 0.025
RF = 0.1281


def sortino_ex_ante(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    port_excess = w @ mu - rf
    asset_excess = mu - rf
    downside_weights = w * (asset_excess < 0)
    sigma_p = np.sqrt(w @ cov @ w)
    sigma_d = (
        np.sqrt(downside_weights @ cov @ downside_weights)
        if downside_weights.sum() > 1e-8
        else sigma_p
    )
    return float(port_excess / sigma_d) if sigma_d > 1e-12 else float("nan")


def markowitz_long_only(mu: np.ndarray, cov: np.ndarray, delta: float, n: int) -> np.ndarray:
    def neg_utility(w: np.ndarray) -> float:
        return float(-(w @ mu - 0.5 * delta * w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Markowitz optimization failed: {res.message}")
    return res.x


def portfolio_metrics(
    name: str,
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    w_mkt: np.ndarray,
    rf: float,
) -> dict:
    mu_p = float(w @ mu)
    sigma = float(np.sqrt(w @ cov @ w))
    sharpe = (mu_p - rf) / sigma if sigma > 0 else float("nan")
    return {
        "portfolio": name,
        "mu_excess_pct": round(mu_p * 100, 2),
        "volatility_pct": round(sigma * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino_ex_ante(w, mu, cov, rf), 3),
        "hhi": round(float(np.sum(w**2)), 3),
        "tracking_error_pct": round(float(np.sqrt((w - w_mkt) @ cov @ (w - w_mkt))) * 100, 2),
        "turnover_vs_market": round(float(np.abs(w - w_mkt).sum() / 2), 3),
        "max_weight_pct": round(float(w.max() * 100), 1),
        "active_assets": int((w > 0.001).sum()),
    }


def main() -> dict:
    cov_df = pd.read_csv(DATA / "ewma_covariance.csv", index_col=0)
    tickers = cov_df.columns.tolist()
    n = len(tickers)
    cov = cov_df.values.astype(float)

    mkt_df = pd.read_csv(DATA / "market_overview.csv", index_col=0).reindex(tickers)
    w_mkt = mkt_df["market_weight"].values.astype(float)
    w_mkt /= w_mkt.sum()
    pi = reverse_optimize(cov, w_mkt, DELTA)

    pred = pd.read_csv(DATA / "analyst_forecasts_2026_with_predictions.csv")
    pred["q_excess"] = pred["potential_earn_rate_ann"] - pred["risk_free_rate_ann"]
    with open(DATA / "model_artifacts/metrics.json", encoding="utf-8") as f:
        ml_metrics = json.load(f)

    views_ml = build_ridge_views(tickers, cov, TAU, pred, ml_metrics["median_y_pred"])
    res_ml = run_black_litterman(tickers, cov, w_mkt, views_ml, delta=DELTA, tau=TAU)

    omega_hl = build_omega_proportional(views_ml.P, cov, TAU)
    views_hl = BLViews(P=views_ml.P, Q=views_ml.Q, Omega=omega_hl, names=views_ml.names)
    res_hl = run_black_litterman(tickers, cov, w_mkt, views_hl, delta=DELTA, tau=TAU)

    w_1n = np.ones(n) / n
    w_mz = markowitz_long_only(pi, cov, DELTA, n)

    w_unconstrained = np.linalg.inv(DELTA * res_ml.cov_bl) @ res_ml.mu_bl

    portfolios = [
        ("BL+ML", res_ml.weights_bl, res_ml.mu_bl),
        ("BL base", res_hl.weights_bl, res_hl.mu_bl),
        ("Market", w_mkt, pi),
        ("1/N", w_1n, pi),
        ("Markowitz", w_mz, pi),
    ]

    benchmark_rows = [
        portfolio_metrics(name, w, mu, cov, w_mkt, RF) for name, w, mu in portfolios
    ]

    forecast_stats = {
        "n_forecasts": len(pred),
        "q_total_mean_pct": round(pred["potential_earn_rate_ann"].mean() * 100, 1),
        "q_total_median_pct": round(pred["potential_earn_rate_ann"].median() * 100, 1),
        "q_excess_mean_pct": round(pred["q_excess"].mean() * 100, 1),
        "q_excess_median_pct": round(pred["q_excess"].median() * 100, 1),
        "q_excess_min_pct": round(pred["q_excess"].min() * 100, 1),
        "q_excess_max_pct": round(pred["q_excess"].max() * 100, 1),
    }

    bl_ml_row = next(r for r in benchmark_rows if r["portfolio"] == "BL+ML")
    market_row = next(r for r in benchmark_rows if r["portfolio"] == "Market")

    summary = {
        "parameters": {"delta": DELTA, "tau": TAU, "rf": RF},
        "note": (
            "Ex-ante метрики на основе μ̂, π и Σ (EWMA). "
            "Max Drawdown требует backtest на реализованных доходностях."
        ),
        "ml_metrics": ml_metrics,
        "forecast_stats": forecast_stats,
        "median_error_pct": round(ml_metrics["median_y_pred"] * 100, 2),
        "unconstrained_weights_pct": {
            "max": round(float(w_unconstrained.max() * 100), 1),
            "min": round(float(w_unconstrained.min() * 100), 1),
        },
        "bl_portfolio_summary": {
            "sharpe_bl": bl_ml_row["sharpe"],
            "sharpe_market": market_row["sharpe"],
            "max_weight_pct": bl_ml_row["max_weight_pct"],
            "max_weight_ticker": tickers[int(res_ml.weights_bl.argmax())],
            "active_assets": bl_ml_row["active_assets"],
            "total_assets": n,
            "hhi": bl_ml_row["hhi"],
            "hhi_market": market_row["hhi"],
            "avg_weight_pct": round(100 / bl_ml_row["active_assets"], 2),
        },
        "benchmarks": benchmark_rows,
    }

    out_path = DATA / "model_artifacts" / "benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {out_path}")
    return summary


if __name__ == "__main__":
    main()
