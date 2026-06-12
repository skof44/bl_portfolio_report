"""
Production Streamlit-отчет для дипломной работы:
"Модель Блэка-Литтермана с ML-калибровкой весов на российском фондовом рынке"

Отчет представляет собой единую длинную страницу с якорной навигацией
в sidebar. Пользователь может переключаться между разделами через ссылки
в боковой панели или листать страницу вниз.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from black_litterman import (
    BLResult,
    DEFAULT_TAU_OMEGA,
    build_ridge_views,
    reverse_optimize,
    run_black_litterman,
)

# ─────────────────────────────────────────────────────────────────────────────
# Соглашение о доходностях (см. black_litterman.py)
# π, μ̂ и Q — избыточные (excess) доходности относительно Rf.
# Total return = excess + Rf (единый Rf из sidebar для всего отчёта).
# ─────────────────────────────────────────────────────────────────────────────


def to_total(mu_excess: np.ndarray, rf: float) -> np.ndarray:
    return mu_excess + rf


def mean_q_excess_per_ticker(
    P: np.ndarray,
    Q: np.ndarray,
    tickers: list[str],
) -> dict[str, float]:
    """Средний excess-Q по тикеру (если на актив несколько views)."""
    n = len(tickers)
    sums = np.zeros(n)
    counts = np.zeros(n)
    for i in range(P.shape[0]):
        idxs = np.where(np.abs(P[i]) > 0.5)[0]
        if len(idxs) == 1:
            j = int(idxs[0])
            sums[j] += Q[i]
            counts[j] += 1
    return {tickers[j]: float(sums[j] / counts[j]) for j in range(n) if counts[j] > 0}


def return_convention_note(rf: float) -> str:
    return (
        f"<b>Соглашение:</b> π, μ̂ и Q в формулах BL — <b>избыточные</b> доходности (excess) "
        f"относительно единого Rf = <b>{rf*100:.2f}%</b>. "
        f"Total = excess + Rf. μ̂ <b>не</b> содержит Rf внутри; если μ̂ excess "
        f"близок к π total — это эффект views, а не ошибка шкалы."
    )


def validate_return_scales(
    pi: np.ndarray,
    mu_bl: np.ndarray,
    rf: float,
    w_mkt: np.ndarray,
) -> dict[str, float]:
    """Числовые проверки согласованности excess/total."""
    pi_total = to_total(pi, rf)
    mu_total = to_total(mu_bl, rf)
    shift_ex = mu_bl - pi
    shift_tot = mu_total - pi_total
    return {
        "market_excess_w_pi_pct": float(w_mkt @ pi * 100),
        "pi_mean_excess_pct": float(pi.mean() * 100),
        "pi_mean_total_pct": float(pi_total.mean() * 100),
        "mu_mean_excess_pct": float(mu_bl.mean() * 100),
        "mu_mean_total_pct": float(mu_total.mean() * 100),
        "max_abs_shift_ex_minus_tot_pct": float(np.max(np.abs(shift_ex - shift_tot)) * 100),
        "max_abs_mu_minus_pi_total_pct": float(np.max(np.abs(mu_bl - pi_total)) * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Шрифты графиков (×2 для скриншотов / вставки в ВКР)
# ─────────────────────────────────────────────────────────────────────────────
CHART_FONT_SCALE = 2.0
_CHART_FONT_FAMILY = "Inter, Arial, sans-serif"


def _chart_fs(base: float) -> int:
    return int(round(base * CHART_FONT_SCALE))


def style_thesis_figure(fig: go.Figure, **layout) -> go.Figure:
    """Крупные подписи осей, легенды и меток — удобно для Word/PDF."""
    body = _chart_fs(12)
    tick = _chart_fs(11)
    title = _chart_fs(14)
    legend = _chart_fs(11)
    hover = _chart_fs(11)
    heatmap_text = _chart_fs(10)

    margin = layout.pop("margin", None) or {}
    margin = {
        "t": margin.get("t", _chart_fs(50)),
        "b": margin.get("b", _chart_fs(55)),
        "l": margin.get("l", _chart_fs(65)),
        "r": margin.get("r", _chart_fs(25)),
    }

    legend_kw = layout.pop("legend", {}) or {}
    legend_kw = {
        **legend_kw,
        "font": {
            **(legend_kw.get("font") or {}),
            "size": legend,
            "family": _CHART_FONT_FAMILY,
        },
    }

    fig.update_layout(
        font=dict(family=_CHART_FONT_FAMILY, size=body, color="#1a1a1a"),
        title_font=dict(size=title, family=_CHART_FONT_FAMILY),
        legend=legend_kw,
        hoverlabel=dict(font_size=hover, font_family=_CHART_FONT_FAMILY),
        margin=margin,
        **layout,
    )
    fig.update_xaxes(
        tickfont=dict(size=tick, family=_CHART_FONT_FAMILY),
        title_font=dict(size=tick, family=_CHART_FONT_FAMILY),
    )
    fig.update_yaxes(
        tickfont=dict(size=tick, family=_CHART_FONT_FAMILY),
        title_font=dict(size=tick, family=_CHART_FONT_FAMILY),
    )
    try:
        fig.update_coloraxes(
            colorbar=dict(
                tickfont=dict(size=tick, family=_CHART_FONT_FAMILY),
                title_font=dict(size=tick, family=_CHART_FONT_FAMILY),
            ),
        )
    except (ValueError, TypeError):
        pass

    fig.update_traces(
        textfont=dict(size=tick, family=_CHART_FONT_FAMILY),
        textfont_size=tick,
    )

    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None:
            continue
        size = getattr(marker, "size", None)
        if isinstance(size, (int, float)) and size > 0:
            marker.size = max(int(size * CHART_FONT_SCALE), 14)
        if getattr(trace, "type", None) == "heatmap" and hasattr(trace, "textfont"):
            trace.textfont = dict(size=heatmap_text, family=_CHART_FONT_FAMILY)

    return fig


def show_thesis_chart(fig: go.Figure, **layout) -> None:
    style_thesis_figure(fig, **layout)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BL + ML | Дипломный отчет",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS: академический стиль + навигация + scroll-behavior
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

    html {
        scroll-behavior: smooth;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'IBM Plex Serif', serif;
    }

    .hero-report {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: #f0f0f0;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    .hero-report h1 {
        font-family: 'IBM Plex Serif', serif;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }

    .hero-report p {
        opacity: 0.85;
        font-size: 1.05rem;
        margin: 0;
    }

    .section-card {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 1.5rem;
    }

    .formula-box {
        background: rgba(128, 128, 128, 0.1);
        border-left: 4px solid #2c5364;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .katex-display {
        background: rgba(128, 128, 128, 0.1);
        border-left: 4px solid #2c5364;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .insight-box {
        background: rgba(23, 162, 184, 0.12);
        border-left: 4px solid #17a2b8;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    .warning-box {
        background: rgba(255, 193, 7, 0.12);
        border-left: 4px solid #ffc107;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* ── Sidebar навигация ── */
    .nav-link {
        display: block;
        padding: 0.4rem 0.6rem;
        margin: 0.15rem 0;
        color: inherit;
        text-decoration: none;
        border-radius: 6px;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }

    .nav-link:hover {
        background: rgba(128, 128, 128, 0.15);
        color: inherit;
        text-decoration: none;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        border-radius: 8px 8px 0 0;
    }

    /* Якорный отступ, чтобы заголовок не прятался под header */
    .section-anchor {
        display: block;
        position: relative;
        top: -80px;
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Пути к данным
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_FILES = [
    "ewma_covariance.csv",
    "market_overview.csv",
    "analyst_forecasts_2026_with_predictions.csv",
    "model_artifacts/metrics.json",
]

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error(f"Не найдены файлы данных: {', '.join(missing)}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Кэшированная загрузка данных
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> dict:
    cov_df = pd.read_csv(DATA_DIR / "ewma_covariance.csv", index_col=0)
    tickers = cov_df.columns.tolist()
    cov_np = cov_df.values.astype(float)

    mkt_df = pd.read_csv(DATA_DIR / "market_overview.csv", index_col=0)
    mkt_df = mkt_df.reindex(tickers)
    w_mkt = mkt_df["market_weight"].values.astype(float)
    w_mkt /= w_mkt.sum()

    corr_df = pd.read_csv(DATA_DIR / "correlations.csv", index_col=0)
    corr_df = corr_df.reindex(index=tickers, columns=tickers)
    corr_np = corr_df.values.astype(float)

    pred_df = pd.read_csv(DATA_DIR / "analyst_forecasts_2026_with_predictions.csv")
    with open(DATA_DIR / "model_artifacts" / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "cov_df": cov_df,
        "cov_np": cov_np,
        "corr_df": corr_df,
        "corr_np": corr_np,
        "tickers": tickers,
        "mkt_df": mkt_df,
        "w_mkt": w_mkt,
        "pred_df": pred_df,
        "metrics": metrics,
        "median_y_pred": metrics["median_y_pred"],
    }


data = load_data()
tickers = data["tickers"]
n = len(tickers)
cov_np = data["cov_np"]
w_mkt = data["w_mkt"]
pred_df = data["pred_df"]
median_y_pred = data["median_y_pred"]
ml_alpha = data["metrics"].get("best_alpha", 10.0)

# ─────────────────────────────────────────────────────────────────────────────
# Боковая панель: якорная навигация
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Навигация по отчету")
    st.markdown(
        """
        <a href="#section-intro" class="nav-link">📖 Введение</a>
        <a href="#section-data" class="nav-link">📁 Исходные данные</a>
        <a href="#section-prior" class="nav-link">1️⃣ Этап 1: Рыночное равновесие</a>
        <a href="#section-views" class="nav-link">2️⃣ Этап 2: Инвесторские прогнозы</a>
        <a href="#section-ml" class="nav-link">3️⃣ Этап 3: ML-модель и Omega</a>
        <a href="#section-master" class="nav-link">4️⃣ Этап 4: Мастер-формула BL</a>
        <a href="#section-weights" class="nav-link">5️⃣ Этап 5: Оптимизация весов</a>
        <a href="#section-results" class="nav-link">6️⃣ Этап 6: Итоговые результаты</a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Параметры модели")
    delta = st.number_input(
        "δ — неприятие риска",
        min_value=0.1,
        value=2.5,
        step=0.1,
        help="Коэффициент неприятия риска. Типичное значение для развивающихся рынков: 2.0–3.0",
    )
    tau = st.number_input(
        "τ — масштаб prior",
        min_value=0.001,
        max_value=1.0,
        value=0.025,
        step=0.005,
        format="%.3f",
        help=(
            "Влияет на мастер-формулу (вес prior vs views). "
            f"Ω калибруется отдельно при фикс. τ_Ω={DEFAULT_TAU_OMEGA}."
        ),
    )
    rf = st.number_input(
        "Rf — безрисковая ставка",
        min_value=0.0,
        max_value=0.30,
        value=0.1281,
        step=0.001,
        format="%.4f",
        help="прогноз безрисковой доходности ОФЗ на 1 год",
    )

    st.markdown("---")
    st.caption(
        f"**Активов:** {n}  \n"
        f"**Прогнозов:** {len(pred_df)}  \n"
        f"**Медианная ошибка ML:** {median_y_pred*100:.2f}%  \n"
        f"**Rf (excess baseline):** {rf*100:.2f}%"
    )

# Q для BL и отображения: excess относительно единого Rf из sidebar
pred_df = pred_df.copy()
pred_df["q_excess_row_rf"] = (
    pred_df["potential_earn_rate_ann"] - pred_df["risk_free_rate_ann"]
)
pred_df["q_excess"] = pred_df["potential_earn_rate_ann"] - rf

# ─────────────────────────────────────────────────────────────────────────────
# Предварительный расчет BL (кэшированный)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_bl(
    tickers_key: tuple,
    cov_bytes: bytes,
    w_bytes: bytes,
    pred_bytes: bytes,
    delta_f: float,
    tau_f: float,
    rf_f: float,
) -> BLResult:
    cov = pickle.loads(cov_bytes)
    w = pickle.loads(w_bytes)
    pred_local = pickle.loads(pred_bytes)
    views = build_ridge_views(
        list(tickers_key),
        cov,
        pred_local,
        median_error=median_y_pred,
        rf=rf_f,
    )
    return run_black_litterman(
        list(tickers_key), cov, w, views, delta=delta_f, tau=tau_f, long_only=True
    )


try:
    bl_result = compute_bl(
        tickers_key=tuple(tickers),
        cov_bytes=pickle.dumps(cov_np),
        w_bytes=pickle.dumps(w_mkt),
        pred_bytes=pickle.dumps(pred_df),
        delta_f=float(delta),
        tau_f=float(tau),
        rf_f=float(rf),
    )
except Exception as exc:
    st.error(f"Ошибка расчета BL: {exc}")
    bl_result = None


# ═════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═════════════════════════════════════════════════════════════════════════════
def plot_bar_comparison(
    categories: list,
    series: dict[str, np.ndarray],
    title: str,
    ytitle: str = "Значение",
    height: int = 450,
) -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=vals,
                marker_color=colors[i % len(colors)],
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="",
        yaxis_title=ytitle,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def plot_waterfall_weights(
    tickers: list,
    w_mkt: np.ndarray,
    w_bl: np.ndarray,
) -> go.Figure:
    delta_w = w_bl - w_mkt
    sorted_idx = np.argsort(delta_w)[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[tickers[i] for i in sorted_idx],
            y=(delta_w[sorted_idx] * 100).round(2),
            marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in delta_w[sorted_idx]],
            text=[f"{v:+.1f}%" for v in (delta_w[sorted_idx] * 100)],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
    fig.update_layout(
        title="Изменение весов: BL − Рынок",
        xaxis_title="",
        yaxis_title="Δ веса, %",
        height=420,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-report">
        <h1>Модель Блэка–Литтермана с ML-калибровкой весов</h1>
        <p>Портфельная оптимизация на российском фондовом рынке с использованием
        прогнозов аналитиков и Ridge-регрессии для оценки неопределенности</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ: ВВЕДЕНИЕ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-intro" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 📖 Введение")

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown(
        """
        <div class="section-card">
        <h4>Проблема классической модели Марковица</h4>
        <p>
        Модель Марковица (1952) заложила фундамент портфельной теории, но на практике
        сталкивается с рядом фундаментальных проблем:
        </p>
        <ul>
            <li><b>Проблема погрешности оценки</b> — использование исторических средних
            в качестве прокси ожидаемой доходности приводит к смещенным оценкам</li>
            <li><b>Максимизация ошибки</b> — оптимизатор эксплуатирует шум в данных,
            создавая нестабильные портфели (Michaud, 1989)</li>
            <li><b>Экстремальные веса</b> — часто возникают короткие позиции и
            концентрация в 1–2 активах</li>
            <li><b>Отсутствие интуитивности</b> — инвестор не может встроить свои
            субъективные ожидания в оптимизацию</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
        <h4>Модель Блэка–Литтермана (1991, 1992)</h4>
        <p>
        Блэк и Литтерман предложили революционный подход: вместо того чтобы
        «гадать» доходности, <b>восстановить их из наблюдаемых рыночных весов</b>
        через процедуру обратной оптимизации.
        </p>
        <p>Ключевые нововведения:</p>
        <ol>
            <li><b>Рыночное равновесие как стартовая точка</b> — prior π выводится
            из капитализационных весов, а не исторических средних</li>
            <li><b>Взгляды инвестора (Views)</b> — субъективные прогнозы встроены
            в байесовскую процедуру через матрицы P, Q, Ω</li>
            <li><b>Стабильность</b> — posterior доходности являются компромиссом
            между рынком и прогнозами, что дает диверсифицированные портфели</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="section-card">
        <h4>Новизна данной работы</h4>
        <p>
        В отличие от классической спецификации Ω (He & Litterman, 1999; Idzorek, 2005),
        в данной работе неопределенность каждого прогноза аналитика <b>предсказывается
        индивидуально</b> с помощью Ridge-регрессии, обученной на исторических ошибках аналитиков.
        </p>
        <ul>
            <li>Целевая переменная: абсолютная ошибка прогноза</li>
            <li>Признаки: срок, прогнозируемая доходность, безрисковая ставка,
            аналитик, тикер</li>
            <li>Модель штрафует агрессивные прогнозы и «плохих» аналитиков</li>
        </ul>
        <hr style="margin: 1rem 0; border: none; border-top: 1px solid rgba(128,128,128,0.25);">
        <p style="font-size: 0.9rem; opacity: 0.75;">
        <b>23 актива</b> портфеля MOEX · <b>65 прогнозов</b> аналитиков ·
        <b>Ridge (α={ml_alpha:g})</b> с One-Hot Encoding · Q = excess return
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-card" style="text-align:center;">
        <h4>Параметры модели</h4>
        <div style="font-size: 2rem; font-weight: 600;">δ = {delta}</div>
        <div style="font-size: 0.9rem; opacity: 0.75;">неприятие риска</div>
        <div style="margin-top: 0.75rem; font-size: 2rem; font-weight: 600;">τ = {tau:.3f}</div>
        <div style="font-size: 0.9rem; opacity: 0.75;">масштаб prior</div>
        <div style="margin-top: 0.75rem; font-size: 2rem; font-weight: 600;">{median_y_pred*100:.2f}%</div>
        <div style="font-size: 0.9rem; opacity: 0.75;">медианная ошибка ML</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("### Общая схема модели")
st.latex(
    r"""
    \underbrace{\boldsymbol{\pi} = \delta \boldsymbol{\Sigma} \mathbf{w}_{mkt}}_{\text{Этап 1: Prior}}
    \quad \longrightarrow \quad
    \underbrace{\mathbf{P}\boldsymbol{\mu} = \mathbf{Q} + \boldsymbol{\varepsilon}}_{\text{Этап 2: Views}}
    \quad \longrightarrow \quad
    \underbrace{\boldsymbol{\Omega} = f(\text{ML})}_{\text{Этап 3: Omega}}
    """
)
st.latex(
    r"""
    \underbrace{\hat{\boldsymbol{\mu}} = \Big[ (\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}'\boldsymbol{\Omega}^{-1}\mathbf{P} \Big]^{-1}
    \Big[ (\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\pi} + \mathbf{P}'\boldsymbol{\Omega}^{-1}\mathbf{Q} \Big]}_{\text{Этап 4: Мастер-формула}}
    \quad \longrightarrow \quad
    \underbrace{\mathbf{w}^* = (\delta\hat{\boldsymbol{\Sigma}})^{-1}\hat{\boldsymbol{\mu}}}_{\text{Этап 5: Веса}}
    """
)


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ: ИСХОДНЫЕ ДАННЫЕ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-data" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 📁 Исходные данные проекта")

st.markdown(
    f"""
    <div class="insight-box">
    <b>Портфель:</b> {n} ликвидных акций, входящих в индекс МосБиржи.
    Ковариационная матрица оценена методом EWMA (span=60, annualized ×252).
    Рыночные веса — по капитализации на дату формирования.
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Состав портфеля")
    port_df = pd.DataFrame(
        {
            "Тикер": tickers,
            "Капитализация, млрд ₽": (data["mkt_df"]["market_cap_bln_rub"].values).round(2),
            "Рыночный вес, %": (w_mkt * 100).round(2),
            "Диаг. Σ (варианс)": np.diag(cov_np).round(4),
            "Волатильность, %": (np.sqrt(np.diag(cov_np)) * 100).round(2),
        }
    )
    st.dataframe(port_df, use_container_width=True, height=460)

with col_right:
    st.subheader("Рыночная капитализация")
    fig_pie = px.pie(
        port_df,
        names="Тикер",
        values="Капитализация, млрд ₽",
        hole=0.40,
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    show_thesis_chart(fig_pie, height=480, showlegend=False, margin=dict(t=15, b=15))

st.subheader("Корреляционная матрица активов")
fig_cov = px.imshow(
    data["corr_np"],
    x=tickers,
    y=tickers,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    text_auto=".2f",
)
show_thesis_chart(fig_cov, height=620, margin=dict(t=20, b=20))

st.caption(
    "На диагонали корреляционной матрицы стоят 1 (совершенная положительная корреляция актива с самим собой). "
    "Вне диагонали — попарные корреляции доходностей."
)

st.subheader("Прогнозы аналитиков (2026)")
display_pred = pred_df[
    [
        "create_dt",
        "ticker",
        "analyst_name",
        "potential_earn_rate_ann",
        "q_excess",
        "q_excess_row_rf",
        "y_pred",
    ]
].copy()
display_pred["potential_earn_rate_ann"] = (display_pred["potential_earn_rate_ann"] * 100).round(1)
display_pred["q_excess"] = (display_pred["q_excess"] * 100).round(1)
display_pred["q_excess_row_rf"] = (display_pred["q_excess_row_rf"] * 100).round(1)
display_pred["y_pred"] = (display_pred["y_pred"] * 100).round(1)
display_pred.columns = [
    "Дата",
    "Тикер",
    "Аналитик",
    "Total (%)",
    f"Q excess (Rf={rf*100:.1f}%)",
    "Q excess (rf строки)",
    "Предсказанная ошибка (%)",
]
st.dataframe(display_pred, use_container_width=True, height=360, hide_index=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Число прогнозов", len(pred_df))
c2.metric("Уникальных аналитиков", pred_df["analyst_name"].nunique())
c3.metric("Уникальных тикеров", pred_df["ticker"].nunique())
c4.metric("Медианная ошибка ML", f"{median_y_pred*100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: РЫНОЧНОЕ РАВНОВЕСИЕ (PRIOR)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-prior" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 1️⃣ Этап 1: Рыночное равновесие (Prior)")

st.markdown(
    """
    <div class="section-card">
    <h4>Суть подхода</h4>
    <p>
    Если рынок находится в равновесии (CAPM), то наблюдаемые капитализационные веса
    являются оптимальными для «среднего» инвестора с параметром неприятия риска δ.
    Тогда равновесные доходности π можно <b>восстановить</b>, а не гадать.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(r"\boldsymbol{\pi} = \delta \boldsymbol{\Sigma} \mathbf{w}_{mkt}")

st.markdown(
    f'<div class="insight-box">{return_convention_note(rf)}</div>',
    unsafe_allow_html=True,
)

pi = bl_result.pi if bl_result is not None else reverse_optimize(cov_np, w_mkt, delta)
pi_total = to_total(pi, rf)

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Равновесные доходности π по активам")
    pi_df = pd.DataFrame(
        {
            "Тикер": tickers,
            "π (excess), %": (pi * 100).round(2),
            "π total, %": (pi_total * 100).round(2),
            "Рыночный вес, %": (w_mkt * 100).round(2),
        }
    ).sort_values("π (excess), %", ascending=False)
    st.dataframe(pi_df, use_container_width=True, height=500)

with col2:
    st.subheader("Гистограмма π")
    fig = px.histogram(
        pi_df,
        x="π (excess), %",
        nbins=12,
        color_discrete_sequence=["#2c5364"],
        opacity=0.85,
    )
    fig.add_vline(
        x=float(pi_df["π (excess), %"].mean()),
        line_dash="dash",
        line_color="red",
        annotation_font_size=_chart_fs(11),
    )
    show_thesis_chart(fig, height=400, showlegend=False)

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Ключевые факты:</b><br>
        • Средняя π = <b>{(pi.mean()*100):.2f}%</b><br>
        • Мин / Макс = <b>{(pi.min()*100):.2f}%</b> … <b>{(pi.max()*100):.2f}%</b><br>
        • SBER (вес {(w_mkt[tickers.index('SBER')]*100):.1f}%) имеет π ≈ <b>{(pi[tickers.index('SBER')]*100):.2f}%</b><br>
        • PLZL (вес {(w_mkt[tickers.index('PLZL')]*100):.1f}%) имеет π ≈ <b>{(pi[tickers.index('PLZL')]*100):.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Валидация: без views веса совпадают с рынком")
w_check = np.linalg.inv(delta * cov_np) @ pi
w_check = w_check / w_check.sum()
max_diff = float(np.abs(w_check - w_mkt).max())
st.metric("max|w − w_mkt|", f"{max_diff:.2e}")
st.caption(
    "При отсутствии views оптимальный портфель совпадает с рынком с точностью до машинного эпсилон. "
    "Это доказывает корректность обратной оптимизации."
)


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: ИНВЕСТОРСКИЕ ПРОГНОЗЫ (VIEWS)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-views" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 2️⃣ Этап 2: Инвесторские прогнозы (Views)")

st.markdown(
    """
    <div class="section-card">
    <p>
    В модели BL каждый прогноз аналитика — это строка линейной системы
    <b>P · μ = Q + ε</b>. В данном проекте используются <b>абсолютные views</b>:
    каждая строка P содержит единицу только для одного тикера.
  Q — <b>избыточная</b> годовая доходность: potential_earn_rate_ann − risk_free_rate_ann.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    r"\mathbf{P}_{k \times n} \cdot \boldsymbol{\mu} = \mathbf{Q}_{k} + \boldsymbol{\varepsilon}, "
    r"\qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, \boldsymbol{\Omega}), "
    r"\quad Q_k = r^{\mathrm{total}}_k - R_f"
)

st.markdown("### Структура матриц")
c1, c2, c3 = st.columns(3)
c1.metric("K (число views)", len(pred_df))
c2.metric("N (число активов)", n)
c3.metric("Размер P", f"{len(pred_df)} × {n}")

st.subheader("Распределение прогнозов по тикерам")
ticker_counts = pred_df["ticker"].value_counts().reset_index()
ticker_counts.columns = ["Тикер", "Число прогнозов"]
fig = px.bar(
    ticker_counts,
    x="Тикер",
    y="Число прогнозов",
    color="Число прогнозов",
    color_continuous_scale="Teal",
    text="Число прогнозов",
)
show_thesis_chart(fig, height=380, showlegend=False)

st.subheader("Распределение прогнозов Q (excess)")
fig2 = go.Figure()
fig2.add_trace(
    go.Histogram(
        x=pred_df["q_excess"] * 100,
        nbinsx=20,
        marker_color="#2c5364",
        opacity=0.8,
    )
)
fig2.add_vline(
    x=float(pred_df["q_excess"].mean() * 100),
    line_dash="dash",
    line_color="red",
    annotation_text=f"μ = {pred_df['q_excess'].mean()*100:.1f}%",
    annotation_font_size=_chart_fs(11),
)
show_thesis_chart(fig2, height=320, xaxis_title="Q excess, % годовых")

st.markdown(
    f"""
    <div class="insight-box">
    <b>Интерпретация:</b><br>
    • Средний Q (excess): <b>{pred_df['q_excess'].mean()*100:.1f}%</b> · total: <b>{pred_df['potential_earn_rate_ann'].mean()*100:.1f}%</b><br>
    • Медианный Q (excess): <b>{pred_df['q_excess'].median()*100:.1f}%</b><br>
    • Максимальный excess: <b>{pred_df['q_excess'].max()*100:.1f}%</b> ({pred_df.loc[pred_df['q_excess'].idxmax(), 'ticker']} / {pred_df.loc[pred_df['q_excess'].idxmax(), 'analyst_name']})<br>
    • Минимальный excess: <b>{pred_df['q_excess'].min()*100:.1f}%</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 3: ML-МОДЕЛЬ И OMEGA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-ml" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 3️⃣ Этап 3: ML-модель и расчёт Omega")

st.markdown(
    """
    <div class="section-card">
    <h4>Зачем нужна ML-калибровка?</h4>
    <p>
    Традиционные подходы к спецификации Ω (He-Litterman, Idzorek) используют
    единую формулу для всех прогнозов. В данной работе предлагается
    <b>индивидуальная оценка неопределенности</b> для каждого прогноза через
    Ridge-регрессию, обученную на исторических ошибках аналитиков.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="section-card">
    <h4>Данные и постановка задачи</h4>
    <p>
    Для обучения использовался исторический датасет <b>SIGNALS_DATA.xlsx</b> —
    {data['metrics']['n_train'] + data['metrics']['n_test']} закрытых рекомендаций аналитиков с известной фактической доходностью.
    Каждая строка — это один прогноз (signal), содержащий дату создания, тикер, аналитика,
    целевую цену, фактическую цену закрытия и макропараметры.
    </p>
    <p><b>Целевая переменная:</b></p>
    <pre style="background:#f4f4f4; padding:0.5rem; border-radius:6px;">
target_y = |fact_earn_rate_ann − potential_earn_rate_ann|
    </pre>
    <p><b>Признаки (5 штук):</b></p>
    <ul>
        <li><b>Числовые (3):</b> term (горизонт прогноза, дней),
            potential_earn_rate_ann (прогнозируемая доходность),
            risk_free_rate_ann (безрисковая ставка на момент прогноза)</li>
        <li><b>Категориальные (2):</b> analyst_name, ticker → One-Hot Encoding</li>
    </ul>
    <p><b>Предобработка:</b></p>
    <ul>
        <li>Удалены {data['metrics'].get('n_removed_zero_target', data['metrics'].get('n_removed_zero', '—'))} сигналов с target_y = 0
            (некорректные данные: идеальное совпадение fact и potential)</li>
        <li>Нормализация названий аналитиков (объединены дубли: «Сбер Инвестиции» → «Сбер», и т.д.)</li>
        <li>Тикеры: топ-30 по частоте + категория «rare_ticker» для остальных</li>
        <li>StandardScaler для числовых признаков</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card">
    <h4>Модель и процедура обучения</h4>
    <p>
    В качестве базового алгоритма выбрана <b>Ridge-регрессия</b> (L2-регуляризация).
    Параметр α подбирается по <b>TimeSeriesSplit</b> на обучающей выборке
    (лучшее значение: <b>α = {ml_alpha:g}</b>).
    </p>
    <ul>
        <li><b>Pipeline:</b> ColumnTransformer (StandardScaler + OHE) → RidgeRegressor</li>
        <li><b>Разбиение:</b> хронологическое 80/20 по create_dt (train: {data['metrics'].get('train_period', {}).get('start', '—')} … {data['metrics'].get('train_period', {}).get('end', '—')})</li>
        <li><b>Тест:</b> {data['metrics'].get('test_period', {}).get('start', '—')} … {data['metrics'].get('test_period', {}).get('end', '—')}</li>
        <li><b>Случайное состояние:</b> random_state = 42 (детерминированность)</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

m = data["metrics"]
c1, c2, c3, c4 = st.columns(4)
_train_mae = m["train_metrics"].get("MAE_%", m["train_metrics"]["MAE"] * 100)
_test_mae = m["test_metrics"].get("MAE_%", m["test_metrics"]["MAE"] * 100)
_test_medae = m["test_metrics"].get("MedAE_%", m["test_metrics"].get("MedAE", 0) * 100)
c1.metric("Train MAE", f"{_train_mae:.2f}%")
c2.metric("Test MAE", f"{_test_mae:.2f}%")
c3.metric("Test R²", f"{m['test_metrics']['R2']:.3f}")
c4.metric("Test MedAE", f"{_test_medae:.2f}%")

st.markdown("---")
st.markdown("### Расчёт Ω в 5 шагов")

# Промежуточные вычисления (те же views, что в BL-расчёте)
if bl_result is not None and bl_result.views is not None:
    views = bl_result.views
else:
    views = build_ridge_views(
        tickers, cov_np, pred_df, median_error=median_y_pred, rf=rf
    )
P, Q = views.P, views.Q
baseline = DEFAULT_TAU_OMEGA * np.diag(P @ cov_np @ P.T)
rel_err = pred_df["y_pred"].values[: views.k] / max(median_y_pred, 1e-8)
omega = np.diag(views.Omega)
ratio = omega / np.maximum(baseline, 1e-12)

step_col1, step_col2 = st.columns(2)

with step_col1:
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Шаг 1:</b> baseline = τ_Ω · diag(PΣP') (τ_Ω = {DEFAULT_TAU_OMEGA:.3f}, фикс.)<br>
        <b>Шаг 2:</b> median_error = {median_y_pred * 100:.2f}%<br>
        <b>Шаг 3:</b> rel_err = predicted_error / median_error<br>
        <b>Шаг 4-5:</b> ω = baseline · (1 + rel_err²) · (1 + 10·Q²)<br>
        <b>τ из sidebar</b> влияет только на этап 4 (мастер-формула), не на Ω.
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_col2:
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Результаты:</b><br>
        • baseline: <b>{baseline.min():.6f}</b> … <b>{baseline.max():.6f}</b><br>
        • omega: <b>{omega.min():.6f}</b> … <b>{omega.max():.6f}</b><br>
        • ratio max = <b>{ratio.max():.2f}</b> (omega в ~{ratio.max():.0f} раз больше baseline)<br>
        • Диагональная Ω: <b>{'Да' if np.allclose(views.Omega, np.diag(np.diag(views.Omega))) else 'Нет'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Распределение omega по прогнозам")
omega_df = pd.DataFrame(
    {
        "Прогноз": views.names,
        "baseline": baseline,
        "omega": omega,
        "ratio": ratio,
        "Q excess (%)": (Q * 100).round(1),
        "rel_err": rel_err.round(2),
    }
)

fig_omega = go.Figure()
fig_omega.add_trace(
    go.Scatter(
        x=omega_df["Q excess (%)"],
        y=omega,
        mode="markers",
        marker=dict(size=8, color="#2c5364"),
        name="omega",
    )
)
show_thesis_chart(
    fig_omega,
    height=400,
    xaxis_title="Q excess (%)",
    yaxis_title="Ω (неопределенность)",
)

st.dataframe(omega_df.sort_values("omega", ascending=False), use_container_width=True, height=320, hide_index=True)

st.markdown(
    """
    <div class="warning-box">
    <b>Ключевой вывод:</b> чем агрессивнее прогноз (высокий Q), тем больше штраф 10·Q² —
    и тем выше omega. Это означает, что BL будет <b>меньше доверять</b> экстремальным прогнозам,
    что предотвращает чрезмерную концентрацию портфеля.
    </div>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 4: МАСТЕР-ФОРМУЛА BL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-master" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 4️⃣ Этап 4: Мастер-формула BL")

st.markdown(
    """
    <div class="section-card">
    <p>
    Мастер-формула объединяет <b>apriori</b> рыночное равновесие (π, τΣ) и
    <b>views</b> инвестора (P, Q, Ω) в байесовском смысле. Результат —
    posterior доходности μ̂ и posterior ковариация Σ̂.
    π и μ̂ — <b>excess</b>-доходности; для сравнения с номинальными ставками
    используйте столбец total = excess + Rf.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="insight-box">{return_convention_note(rf)}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="warning-box">
    <b>Два параметра τ:</b> слайдер <b>τ = {tau:.3f}</b> в sidebar управляет только
    мастер-формулой (компромисс prior π vs views). Калибровка Ω на этапе 3 использует
    фиксированный <b>τ_Ω = {DEFAULT_TAU_OMEGA:.3f}</b> — при изменении τ в UI матрица Ω
    не пересчитывается, меняются только μ̂ и Σ̂.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Long-form формула")
st.latex(
    r"""
    \mathbf{M} = (\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}'\boldsymbol{\Omega}^{-1}\mathbf{P}
    """
)
st.latex(
    r"""
    \hat{\boldsymbol{\mu}} = \mathbf{M}^{-1} \Big[ (\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\pi} + \mathbf{P}'\boldsymbol{\Omega}^{-1}\mathbf{Q} \Big]
    """
)
st.latex(
    r"""
    \hat{\boldsymbol{\Sigma}} = \mathbf{M}^{-1} + \boldsymbol{\Sigma}
    """
)

st.markdown("### Short-form (эквивалентная)")
st.latex(
    r"""
    \hat{\boldsymbol{\mu}} = \boldsymbol{\pi} + \tau\boldsymbol{\Sigma}\mathbf{P}' \Big[ \tau\mathbf{P}\boldsymbol{\Sigma}\mathbf{P}' + \boldsymbol{\Omega} \Big]^{-1} (\mathbf{Q} - \mathbf{P}\boldsymbol{\pi})
    """
)

if bl_result is None:
    st.error("BL-результат недоступен.")
else:
    # Проверка эквивалентности
    pi = bl_result.pi
    mu_bl = bl_result.mu_bl
    pi_total = to_total(pi, rf)
    mu_total = to_total(mu_bl, rf)
    views = bl_result.views
    tau_cov = tau * cov_np
    A = tau_cov @ views.P.T @ np.linalg.inv(tau * views.P @ cov_np @ views.P.T + views.Omega)
    mu_alt = pi + A @ (views.Q - views.P @ pi)
    diff = float(np.max(np.abs(mu_bl - mu_alt)))

    st.metric("Разница long-form vs short-form", f"{diff:.2e}", help="Должна быть << 1e-8")

    checks = validate_return_scales(pi, mu_bl, rf, w_mkt)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("w·π (excess рынка)", f"{checks['market_excess_w_pi_pct']:.2f}%")
    v2.metric("Средняя π excess", f"{checks['pi_mean_excess_pct']:.2f}%")
    v3.metric("Средняя μ̂ excess", f"{checks['mu_mean_excess_pct']:.2f}%")
    v4.metric("Средняя π total", f"{checks['pi_mean_total_pct']:.2f}%")
    st.caption(
        f"Проверка шкал: max|Δ_excess − Δ_total| = {checks['max_abs_shift_ex_minus_tot_pct']:.2e} п.п. "
        f"(должно быть ≈ 0). μ̂ excess ≠ π total: "
        f"средняя μ̂ total = {checks['mu_mean_total_pct']:.2f}%."
    )

    tab_tot, tab_ex = st.tabs(["Total (excess + Rf)", "Excess (BL-формулы)"])

    with tab_tot:
        st.subheader("Сравнение total-доходностей")
        fig_tot = plot_bar_comparison(
            tickers,
            {
                "π total, %": np.round(pi_total * 100, 2),
                "μ̂ total, %": np.round(mu_total * 100, 2),
            },
            "π vs μ̂ (total)",
            ytitle="Total, % годовых",
        )
        show_thesis_chart(fig_tot)

    with tab_ex:
        st.subheader("Сравнение excess-доходностей")
        fig_ex = plot_bar_comparison(
            tickers,
            {
                "π excess, %": np.round(pi * 100, 2),
                "μ̂ excess, %": np.round(mu_bl * 100, 2),
            },
            "π vs μ̂ (excess)",
            ytitle="Excess, % годовых",
        )
        show_thesis_chart(fig_ex)

    st.subheader("Сдвиг доходностей: μ̂ − π")
    shift_excess = (mu_bl - pi) * 100
    shift_total = (mu_total - pi_total) * 100
    shift_df = pd.DataFrame(
        {
            "Тикер": tickers,
            "π excess, %": np.round(pi * 100, 2),
            "μ̂ excess, %": np.round(mu_bl * 100, 2),
            "Δ excess (μ̂−π), %": np.round(shift_excess, 2),
            "π total, %": np.round(pi_total * 100, 2),
            "μ̂ total, %": np.round(mu_total * 100, 2),
            "Δ total, %": np.round(shift_total, 2),
        }
    ).sort_values("Δ excess (μ̂−π), %", ascending=False)
    fig_shift = go.Figure(
        go.Bar(
            x=shift_df["Тикер"],
            y=shift_df["Δ excess (μ̂−π), %"],
            marker_color=[
                "#2ca02c" if v >= 0 else "#d62728"
                for v in shift_df["Δ excess (μ̂−π), %"]
            ],
        )
    )
    show_thesis_chart(fig_shift, height=380, yaxis_title="Δ excess, п.п.")
    st.dataframe(shift_df, use_container_width=True, height=420, hide_index=True)
    st.caption(
        "Столбцы Δ excess и Δ total совпадают: (μ̂+Rf)−(π+Rf) = μ̂−π. "
        "μ̂ excess может быть сопоставим по величине с π total (~"
        f"{checks['pi_mean_total_pct']:.1f}% vs ~{checks['mu_mean_excess_pct']:.1f}%), "
        "потому что views тянут μ̂ к Q excess ~24%, а не потому что в μ̂ «зашит» Rf."
    )

    # Проверка: posterior ближе к равновесию, чем сырой Q?
    Q_map = mean_q_excess_per_ticker(views.P, views.Q, tickers)
    abs_post = np.abs(mu_bl - pi)
    abs_Q = np.zeros(n)
    for i, t in enumerate(tickers):
        if t in Q_map:
            abs_Q[i] = abs(Q_map[t] - pi[i])
    closer = (abs_post < abs_Q).sum()
    total_with_views = sum(1 for t in tickers if t in Q_map)
    if total_with_views > 0:
        closer_pct = closer / total_with_views * 100
        st.metric(
            "Активов, у которых posterior ближе к рынку",
            f"{closer}/{total_with_views} ({closer_pct:.0f}%)",
        )
        st.caption(
            "Для большинства активов BL «сглаживает» экстремальные прогнозы аналитиков, "
            "приближая апостериорную доходность к равновесной."
        )

    st.subheader("Карта риск–доходность")
    vols = np.sqrt(np.diag(cov_np)) * 100
    rr_tab_tot, rr_tab_ex = st.tabs(["Total (+ Rf)", "Excess"])
    with rr_tab_ex:
        fig_rr_ex = go.Figure()
        fig_rr_ex.add_trace(
            go.Scatter(
                x=vols,
                y=np.round(pi * 100, 2),
                mode="markers+text",
                text=tickers,
                textposition="top center",
                marker=dict(size=10, color="#1f77b4"),
                name="π excess",
            )
        )
        fig_rr_ex.add_trace(
            go.Scatter(
                x=vols,
                y=np.round(mu_bl * 100, 2),
                mode="markers+text",
                text=tickers,
                textposition="bottom center",
                marker=dict(size=10, color="#2ca02c", symbol="diamond"),
                name="μ̂ excess",
            )
        )
        show_thesis_chart(
            fig_rr_ex,
            height=480,
            xaxis_title="Годовая волатильность, %",
            yaxis_title="Excess-доходность, %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
    with rr_tab_tot:
        fig_rr = go.Figure()
        fig_rr.add_trace(
            go.Scatter(
                x=vols,
                y=np.round(pi_total * 100, 2),
                mode="markers+text",
                text=tickers,
                textposition="top center",
                marker=dict(size=10, color="#1f77b4"),
                name="π total",
            )
        )
        fig_rr.add_trace(
            go.Scatter(
                x=vols,
                y=np.round(mu_total * 100, 2),
                mode="markers+text",
                text=tickers,
                textposition="bottom center",
                marker=dict(size=10, color="#2ca02c", symbol="diamond"),
                name="μ̂ total",
            )
        )
        show_thesis_chart(
            fig_rr,
            height=480,
            xaxis_title="Годовая волатильность, %",
            yaxis_title="Total-доходность, %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 5: ОПТИМИЗАЦИЯ ВЕСОВ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-weights" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 5️⃣ Этап 5: Оптимизация весов")

st.markdown(
    """
    <div class="section-card">
    <p>
    Оптимальные веса получаются аналитически из задачи Марковица с posterior
    доходностями. В данной работе применяется <b>long-only constraint</b>:
    отрицательные веса обнуляются, после чего производится нормировка.
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    r"\mathbf{w}^* = (\delta\hat{\boldsymbol{\Sigma}})^{-1} \hat{\boldsymbol{\mu}} "
    r"\quad \longrightarrow \quad \text{long-only + normalize}"
)

if bl_result is None:
    st.error("BL-результат недоступен.")
else:
    pi = bl_result.pi
    mu_bl = bl_result.mu_bl
    cov_bl = bl_result.cov_bl
    weights_mkt = bl_result.weights_mkt
    weights_bl = bl_result.weights_bl
    pi_total = to_total(pi, rf)
    mu_total = to_total(mu_bl, rf)

    # Unconstrained (posterior Σ̂) для демонстрации
    w_unconstrained = np.linalg.inv(delta * cov_bl) @ mu_bl

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Веса: рынок vs BL")
        fig = plot_bar_comparison(
            tickers,
            {
                "Рынок, %": np.round(weights_mkt * 100, 2),
                "BL long-only, %": np.round(weights_bl * 100, 2),
            },
            "Сравнение весов портфелей",
            ytitle="Вес, %",
        )
        show_thesis_chart(fig)

    with col2:
        st.subheader("Изменение весов (waterfall)")
        fig_wf = plot_waterfall_weights(tickers, weights_mkt, weights_bl)
        show_thesis_chart(fig_wf)

    st.subheader("Сводная таблица")
    res_df = pd.DataFrame(
        {
            "Тикер": tickers,
            "π (excess), %": np.round(pi * 100, 2),
            "π total, %": np.round(pi_total * 100, 2),
            "μ̂ excess, %": np.round(mu_bl * 100, 2),
            "μ̂ total, %": np.round(mu_total * 100, 2),
            "Δ excess (μ̂−π), %": np.round((mu_bl - pi) * 100, 2),
            "w_mkt, %": np.round(weights_mkt * 100, 2),
            "w_BL, %": np.round(weights_bl * 100, 2),
            "Δw (BL−mkt), %": np.round((weights_bl - weights_mkt) * 100, 2),
        }
    ).sort_values("w_BL, %", ascending=False)
    st.dataframe(res_df, use_container_width=True, height=520)

    st.subheader("Контроль «здравого смысла»")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Макс. unconstrained вес", f"{w_unconstrained.max()*100:.1f}%")
    c2.metric("Мин. unconstrained вес", f"{w_unconstrained.min()*100:.1f}%")
    c3.metric("Макс. long-only вес", f"{weights_bl.max()*100:.1f}%")
    c4.metric("Активов с w > 0", f"{(weights_bl > 0.001).sum()}/{n}")

    st.markdown(
        f"""
        <div class="warning-box">
        <b>Без ограничений</b> Марковиц давал бы экстремальные веса:
        max = <b>{w_unconstrained.max()*100:.0f}%</b>, min = <b>{w_unconstrained.min()*100:.0f}%</b>.<br>
        <b>Long-only constraint</b> обрезает short-позиции и нормирует, давая
        реалистичный портфель: max = <b>{weights_bl.max()*100:.1f}%</b> (SBER), min = 0%.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 6: ИТОГОВЫЕ РЕЗУЛЬТАТЫ
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<span id="section-results" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown("## 6️⃣ Этап 6: Итоговые результаты портфеля")

if bl_result is None:
    st.error("BL-результат недоступен.")
else:
    mu_bl = bl_result.mu_bl
    cov_bl = bl_result.cov_bl
    mu_total = to_total(mu_bl, rf)
    weights_bl = bl_result.weights_bl
    weights_mkt = bl_result.weights_mkt

    mu_p = weights_bl @ mu_bl
    mu_p_total = mu_p + rf
    sigma_p = np.sqrt(weights_bl @ cov_bl @ weights_bl)
    sharpe_bl = (mu_p - rf) / sigma_p

    mu_p_mkt = weights_mkt @ bl_result.pi
    sigma_p_mkt = np.sqrt(weights_mkt @ cov_np @ weights_mkt)
    sharpe_mkt = (mu_p_mkt - rf) / sigma_p_mkt

    te = np.sqrt((weights_bl - weights_mkt) @ cov_np @ (weights_bl - weights_mkt))

    st.markdown("### Ключевые метрики портфеля")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Изб. доходность BL",
        f"{mu_p*100:.2f}%",
        delta=f"vs {mu_p_mkt*100:.2f}% рынок",
        help="Excess return портфеля (без Rf в числителе Sharpe)",
    )
    m2.metric("Волатильность BL", f"{sigma_p*100:.2f}%", delta=f"vs {sigma_p_mkt*100:.2f}% рынок")
    m3.metric("Sharpe BL", f"{sharpe_bl:.3f}", delta=f"vs {sharpe_mkt:.3f} рынок")
    m4.metric("Tracking Error", f"{te*100:.2f}%")
    m5.metric("Активов в портфеле", f"{(weights_bl > 0.001).sum()}/{n}")
    st.caption(
        f"Total-доходность BL-портфеля ≈ {mu_p_total*100:.2f}% "
        f"(изб. {mu_p*100:.2f}% + Rf {rf*100:.2f}%). "
        f"Волатильность и Sharpe BL — на апостериорной Σ̂; рынок — на prior Σ; "
        f"tracking error — на prior Σ (EWMA)."
    )

    st.markdown("---")
    st.subheader("Структура итогового портфеля")

    final_df = pd.DataFrame(
        {
            "Тикер": tickers,
            "Вес BL, %": np.round(weights_bl * 100, 2),
            "Вес рынок, %": np.round(weights_mkt * 100, 2),
            "Отклонение, п.п.": np.round((weights_bl - weights_mkt) * 100, 2),
            "Вклад (excess), п.п.": np.round(weights_bl * mu_bl * 100, 2),
            "Вклад (total), п.п.": np.round(weights_bl * mu_total * 100, 2),
        }
    ).sort_values("Вес BL, %", ascending=False)

    fig_pie_bl = px.pie(
        final_df[final_df["Вес BL, %"] > 0],
        names="Тикер",
        values="Вес BL, %",
        hole=0.35,
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_pie_bl.update_traces(textposition="inside", textinfo="percent+label")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.dataframe(final_df, use_container_width=True, height=520)
    with col_right:
        show_thesis_chart(fig_pie_bl, height=500, showlegend=False, margin=dict(t=15, b=15))

    st.markdown("---")
    st.subheader("Сравнение портфелей (ex-ante)")

    benchmarks_path = DATA_DIR / "model_artifacts" / "benchmarks.json"
    if benchmarks_path.exists():
        with open(benchmarks_path, encoding="utf-8") as f:
            bench = json.load(f)
        bench_df = pd.DataFrame(bench["benchmarks"]).set_index("portfolio").T
        bench_df.index = [
            "Изб. доходность, %",
            "Волатильность, %",
            "Sharpe",
            "Sortino",
            "HHI",
            "Tracking Error, %",
            "Turnover vs рынок",
            "Max вес, %",
            "Активов с w>0",
        ]
        st.dataframe(bench_df, use_container_width=True)
        st.caption(
            "Полная таблица: `python scripts/compute_benchmarks.py` · "
            "см. docs/thesis-tables.md"
        )
    else:
        comparison = pd.DataFrame(
            {
                "Метрика": [
                    "Избыточная доходность, %",
                    "Волатильность, %",
                    "Sharpe ratio",
                    "Max вес, %",
                    "Активов с w>0",
                ],
                "BL-портфель": [
                    f"{mu_p*100:.2f}",
                    f"{sigma_p*100:.2f}",
                    f"{sharpe_bl:.3f}",
                    f"{weights_bl.max()*100:.1f}",
                    f"{(weights_bl > 0.001).sum()}",
                ],
                "Рыночный портфель": [
                    f"{mu_p_mkt*100:.2f}",
                    f"{sigma_p_mkt*100:.2f}",
                    f"{sharpe_mkt:.3f}",
                    f"{weights_mkt.max()*100:.1f}",
                    f"{n}",
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    max_ticker = tickers[int(weights_bl.argmax())]
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Выводы (ex-ante):</b><br>
        • BL+ML: Sharpe = <b>{sharpe_bl:.3f}</b> vs рынок <b>{sharpe_mkt:.3f}</b>.<br>
        • Tracking Error = <b>{te*100:.2f}%</b> · HHI = <b>{float(np.sum(weights_bl**2)):.3f}</b>.<br>
        • Макс. вес = <b>{weights_bl.max()*100:.1f}%</b> ({max_ticker}) · активов: <b>{(weights_bl > 0.001).sum()}/{n}</b>.<br>
        • ML-Ω снижает концентрацию и TE относительно BL base, но Sharpe ниже базовой BL.
        </div>
        """,
        unsafe_allow_html=True,
    )


