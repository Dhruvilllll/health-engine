"""
Health Engine - Gradio Dashboard

Dark readiness dashboard for manual check-ins, model predictions, and
historical physiological trends.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config.config import config
from src.common.utils import handle_missing_values, standardize_columns
from src.services.batch_service import BatchService
from src.services.prediction_service import PredictionService


prediction_service = PredictionService()
batch_service = BatchService(prediction_service)


DISPLAY_NAMES = {
    "cycle_start_time": "Date",
    "recovery_score_": "Recovery Score %",
    "heart_rate_variability_ms": "HRV (ms)",
    "resting_heart_rate_bpm": "Resting HR (bpm)",
    "sleep_performance_": "Sleep Performance %",
    "day_strain": "Day Strain",
    "blood_oxygen_": "Blood Oxygen %",
    "average_hr_bpm": "Average HR (bpm)",
    "stress_ratio": "Stress Ratio",
    "hrv_strain_balance": "HRV Strain Balance",
    "hrv_7d_avg": "HRV 7D Avg",
    "strain_7d_avg": "Strain 7D Avg",
}

NUMERIC_HISTORY_COLUMNS = [
    "recovery_score_",
    "heart_rate_variability_ms",
    "resting_heart_rate_bpm",
    "sleep_performance_",
    "day_strain",
    "blood_oxygen_",
    "average_hr_bpm",
    "stress_ratio",
    "hrv_strain_balance",
    "hrv_7d_avg",
    "strain_7d_avg",
]

TREND_SPECS = [
    ("recovery_score_", "Recovery Score %", "#35d07f"),
    ("heart_rate_variability_ms", "HRV (ms)", "#4da8ff"),
    ("resting_heart_rate_bpm", "Resting Heart Rate (bpm)", "#ff5a49"),
    ("day_strain", "Day Strain", "#ffb020"),
    ("sleep_performance_", "Sleep Performance %", "#a366ff"),
    ("blood_oxygen_", "Blood Oxygen %", "#27c8b9"),
]

PLOT_BACKGROUND = "#0f141d"
PAGE_BACKGROUND = "#090d14"
GRID_COLOR = "rgba(255,255,255,0.08)"
TEXT_PRIMARY = "#ecf3fb"
TEXT_SECONDARY = "#98a8bb"

CUSTOM_CSS = """
:root {
  --page-bg: #090d14;
  --panel-bg: #121824;
  --surface-bg: #0f141d;
  --surface-border: #1d2635;
  --text-primary: #ecf3fb;
  --text-secondary: #98a8bb;
  --accent-green: #35d07f;
  --accent-blue: #4da8ff;
  --accent-purple: #a366ff;
}

body, .gradio-container {
  background: var(--page-bg) !important;
  color: var(--text-primary) !important;
}

.gradio-container {
  max-width: 100% !important;
}

footer {
  display: none !important;
}

#app-shell {
  gap: 0 !important;
}

.sidebar-shell {
  background: var(--panel-bg);
  border-right: 1px solid var(--surface-border);
  min-height: 100vh;
  padding: 20px 18px 26px !important;
}

.main-shell {
  padding: 22px 26px 32px !important;
}

.sidebar-title {
  font-size: 19px;
  font-weight: 700;
  margin-bottom: 6px;
}

.sidebar-copy {
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
  margin-bottom: 18px;
}

.page-title {
  font-size: 30px;
  font-weight: 700;
  margin-bottom: 4px;
}

.page-copy {
  color: var(--text-secondary);
  font-size: 13px;
  margin-bottom: 16px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
  margin: 12px 0 18px;
}

.metric-tile {
  background: var(--surface-bg);
  border: 1px solid var(--surface-border);
  border-radius: 8px;
  padding: 14px 16px;
  min-height: 96px;
}

.metric-label {
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0;
  text-transform: uppercase;
}

.metric-value {
  display: block;
  font-size: 34px;
  line-height: 1.1;
  font-weight: 700;
  margin-top: 10px;
}

.metric-subtext {
  display: block;
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 8px;
}

.status-banner {
  border-radius: 8px;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 700;
  margin-bottom: 20px;
}

.status-banner span:last-child {
  font-weight: 600;
  opacity: 0.95;
}

.status-poor {
  background: #4d1e25;
  color: #ffd7dc;
}

.status-moderate {
  background: #5a4214;
  color: #ffefbf;
}

.status-good {
  background: #174734;
  color: #d6ffea;
}

.status-excellent {
  background: #1f6d42;
  color: #f0fff6;
}

.section-label {
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.recommendation-shell {
  background: var(--surface-bg);
  border: 1px solid var(--surface-border);
  border-radius: 8px;
  padding: 18px 20px;
}

.recommendation-title {
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 8px;
}

.recommendation-copy {
  color: var(--text-secondary);
  line-height: 1.6;
  margin-bottom: 14px;
}

.recommendation-shell ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-primary);
}

.recommendation-shell li {
  margin-bottom: 8px;
}

.gradio-tabs {
  border-top: 1px solid var(--surface-border);
  margin-top: 8px;
  padding-top: 12px;
}

.block.gradio-plot,
.block.gradio-html,
.block.gradio-dataframe,
.block.gradio-textbox {
  background: transparent !important;
  border: 0 !important;
}

.block label,
.block .wrap > label,
.form label {
  color: var(--text-primary) !important;
}

.gradio-slider input,
.gradio-slider .wrap,
.gradio-slider .container {
  background: transparent !important;
}

.gradio-button.primary {
  background: #1f6d42 !important;
}

@media (max-width: 1280px) {
  .summary-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .sidebar-shell {
    min-height: auto;
  }
}
"""


def find_server_port(default_port: int = 7860) -> int:
    """Use the requested Gradio port, or find a nearby free port locally."""
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        return int(configured_port)

    for port in range(default_port, default_port + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    raise OSError(f"No free Gradio port found in range {default_port}-{default_port + 39}")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_dashboard_history() -> pd.DataFrame:
    """Load the best available physiological history for the dashboard."""
    candidates = [
        config.PHYSIOLOGICAL_CLEANED,
        config.PHYSIOLOGICAL_MERGED,
        Path("data/raw/physiological/physiological_cycles.csv"),
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            df = standardize_columns(df)
            df = prediction_service._prepare_data(df.copy())

            for column in NUMERIC_HISTORY_COLUMNS:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")

            if "cycle_start_time" in df.columns:
                df["cycle_start_time"] = pd.to_datetime(df["cycle_start_time"], errors="coerce")
            else:
                df["cycle_start_time"] = pd.date_range(
                    end=pd.Timestamp.today().normalize(),
                    periods=len(df),
                    freq="D",
                )

            df = handle_missing_values(df, method="median")
            df = df.sort_values("cycle_start_time").reset_index(drop=True)
            return df

    raise FileNotFoundError("No physiological dataset found for dashboard rendering.")


HISTORY_DF = load_dashboard_history()


def latest_or_default(column: str, default: float) -> float:
    """Get the latest value from history or use a fallback default."""
    if column in HISTORY_DF.columns:
        series = pd.to_numeric(HISTORY_DF[column], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return default


def latest_input_values():
    """Return slider defaults from the newest available record."""
    return (
        round(latest_or_default("day_strain", 8.0), 1),
        round(latest_or_default("resting_heart_rate_bpm", 60.0), 1),
        round(latest_or_default("heart_rate_variability_ms", 140.0), 1),
        round(latest_or_default("sleep_performance_", 78.0), 1),
        round(latest_or_default("blood_oxygen_", 97.5), 2),
        round(latest_or_default("average_hr_bpm", 78.0), 1),
    )


def classify_status(score: float) -> str:
    """Classify readiness score."""
    return PredictionService._classify_recovery(score)


def score_metric(value: float, series: pd.Series, reverse: bool = False) -> float:
    """Scale a metric into a 0-100 readiness contribution band."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 50.0

    lower = float(clean.quantile(0.05))
    upper = float(clean.quantile(0.95))
    if np.isclose(lower, upper):
        return 50.0

    clipped = float(np.clip(value, lower, upper))
    scaled = 100 * (clipped - lower) / (upper - lower)
    if reverse:
        scaled = 100 - scaled

    return float(np.clip(scaled, 0, 100))


def calculate_recovery_score(
    heart_rate_variability_ms: float,
    resting_heart_rate_bpm: float,
    day_strain: float,
    blood_oxygen: float,
    sleep_performance: float,
) -> tuple[float, dict[str, float]]:
    """Calculate a dashboard recovery score from current manual inputs."""
    weighted_parts = {
        "Sleep": 0.40 * score_metric(sleep_performance, HISTORY_DF["sleep_performance_"]),
        "HRV": 0.30 * score_metric(heart_rate_variability_ms, HISTORY_DF["heart_rate_variability_ms"]),
        "Resting HR": 0.20 * score_metric(
            resting_heart_rate_bpm,
            HISTORY_DF["resting_heart_rate_bpm"],
            reverse=True,
        ),
        "Day Strain": 0.10 * score_metric(day_strain, HISTORY_DF["day_strain"], reverse=True),
        "SpO2": 0.05 * score_metric(blood_oxygen, HISTORY_DF["blood_oxygen_"]),
    }
    total_weight = 1.05
    score = sum(weighted_parts.values()) / total_weight
    return round(float(np.clip(score, 0, 100)), 1), weighted_parts


def signal_band(value: float, series: pd.Series, labels: tuple[str, str, str]) -> str:
    """Convert a value into a low/mid/high label using historical tertiles."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return labels[1]

    low = float(clean.quantile(0.33))
    high = float(clean.quantile(0.66))

    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def build_summary_html(
    predicted_score: float,
    recovery_score: float,
    overall_score: float,
    status: str,
    hrv_signal: str,
    sleep_signal: str,
) -> str:
    """Render the top summary tiles."""
    return f"""
    <div class="summary-grid">
      <div class="metric-tile">
        <div class="metric-label">Predicted Score</div>
        <span class="metric-value">{predicted_score:.0f}/100</span>
        <span class="metric-subtext">Model output</span>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Recovery Score</div>
        <span class="metric-value">{recovery_score:.0f}/100</span>
        <span class="metric-subtext">Physiology weighted</span>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Status</div>
        <span class="metric-value" style="font-size:26px;">{status}</span>
        <span class="metric-subtext">Overall readiness</span>
      </div>
      <div class="metric-tile">
        <div class="metric-label">HRV Signal</div>
        <span class="metric-value" style="font-size:26px;">{hrv_signal}</span>
        <span class="metric-subtext">Compared with 30-day history</span>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Sleep Signal</div>
        <span class="metric-value" style="font-size:26px;">{sleep_signal}</span>
        <span class="metric-subtext">Current recovery driver</span>
      </div>
    </div>
    """


def build_status_banner(status: str, overall_score: float) -> str:
    """Render the readiness banner."""
    css_class = {
        "Poor Recovery": "status-poor",
        "Moderate Recovery": "status-moderate",
        "Good Recovery": "status-good",
        "Excellent Recovery": "status-excellent",
    }.get(status, "status-good")

    return (
        f'<div class="status-banner {css_class}">'
        f"<span>{status}</span>"
        f"<span>Readiness Score: {overall_score:.0f}/100</span>"
        "</div>"
    )


def make_gauge_figure(title: str, value: float, color: str) -> go.Figure:
    """Build a compact dark gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16, "color": TEXT_PRIMARY}},
            number={"suffix": "/100", "font": {"size": 34, "color": color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#92a4ba",
                    "tickfont": {"color": "#92a4ba"},
                },
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": PLOT_BACKGROUND,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255, 112, 112, 0.22)"},
                    {"range": [40, 60], "color": "rgba(255, 199, 102, 0.20)"},
                    {"range": [60, 80], "color": "rgba(77, 168, 255, 0.18)"},
                    {"range": [80, 100], "color": "rgba(84, 222, 140, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#e8f2ff", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=28, r=28, t=48, b=12),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=TEXT_PRIMARY, family="Arial"),
    )
    return fig


def with_today_marker(column: str, current_value: float | None = None) -> pd.DataFrame:
    """Return history with an optional appended point for the current manual value."""
    frame = HISTORY_DF[["cycle_start_time", column]].copy()
    frame["cycle_start_time"] = pd.to_datetime(frame["cycle_start_time"], errors="coerce")
    frame = frame.dropna(subset=["cycle_start_time"])
    frame = frame.tail(29)

    if current_value is not None:
        today_row = pd.DataFrame(
            {"cycle_start_time": [pd.Timestamp.today().normalize()], column: [current_value]}
        )
        frame = pd.concat([frame, today_row], ignore_index=True)

    frame["label"] = frame["cycle_start_time"].dt.strftime("%b %d")
    return frame


def make_trend_figure(
    column: str,
    title: str,
    color: str,
    current_value: float | None = None,
) -> go.Figure:
    """Render a small trend chart in the dashboard style."""
    frame = with_today_marker(column, current_value)
    values = pd.to_numeric(frame[column], errors="coerce")
    rolling = values.rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fill_color = color if color.startswith("rgba") else None
    if fill_color is None:
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        fill_color = f"rgba({red}, {green}, {blue}, 0.20)"

    fig.add_trace(
        go.Scatter(
            x=frame["label"],
            y=values,
            mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"{title}: %{{y:.1f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["label"],
            y=rolling,
            mode="lines",
            line=dict(color="rgba(230,240,255,0.45)", width=1.6, dash="dash"),
            hoverinfo="skip",
        )
    )

    if current_value is not None:
        fig.add_trace(
            go.Scatter(
                x=[frame["label"].iloc[-1]],
                y=[current_value],
                mode="markers",
                marker=dict(size=8, color="#ffffff", line=dict(color=color, width=2)),
                hovertemplate=f"Today: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.98, xanchor="right", font=dict(size=13, color=TEXT_PRIMARY)),
        height=200,
        margin=dict(l=12, r=12, t=32, b=20),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=TEXT_PRIMARY, family="Arial"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#a3b4c7")),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


def make_feature_importance_figure() -> go.Figure:
    """Render trained model feature importances."""
    features = list(getattr(prediction_service.model, "feature_names_in_", []))
    importances = list(getattr(prediction_service.model, "feature_importances_", []))

    frame = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance")
    frame["label"] = frame["feature"].map(lambda value: DISPLAY_NAMES.get(value, value.replace("_", " ").title()))

    fig = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=frame["label"],
            orientation="h",
            marker=dict(color="#4da8ff"),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Model Feature Importance", x=0.02, font=dict(size=16, color=TEXT_PRIMARY)),
        height=320,
        margin=dict(l=20, r=20, t=48, b=20),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=TEXT_PRIMARY, family="Arial"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def make_signal_strength_figure(contributions: dict[str, float]) -> go.Figure:
    """Render current signal contributions to the readiness score."""
    frame = (
        pd.DataFrame({"signal": list(contributions.keys()), "score": list(contributions.values())})
        .sort_values("score")
        .reset_index(drop=True)
    )

    fig = go.Figure(
        go.Bar(
            x=frame["score"],
            y=frame["signal"],
            orientation="h",
            marker=dict(color=["#a366ff", "#27c8b9", "#ffb020", "#ff5a49", "#35d07f"]),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Current Signal Strength", x=0.02, font=dict(size=16, color=TEXT_PRIMARY)),
        height=320,
        margin=dict(l=20, r=20, t=48, b=20),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color=TEXT_PRIMARY, family="Arial"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, range=[0, 40]),
        yaxis=dict(showgrid=False),
    )
    return fig


def build_recommendation_html(
    status: str,
    overall_score: float,
    predicted_score: float,
    recovery_score: float,
    recommendation: str,
    hrv_signal: str,
    sleep_signal: str,
) -> str:
    """Build recommendation content for the dashboard."""
    return f"""
    <div class="recommendation-shell">
      <div class="section-label">Readiness Outlook</div>
      <div class="recommendation-title">{recommendation}</div>
      <div class="recommendation-copy">
        Overall readiness is <strong>{overall_score:.0f}/100</strong> with a model prediction of
        <strong>{predicted_score:.0f}/100</strong> and a physiology-weighted recovery score of
        <strong>{recovery_score:.0f}/100</strong>.
      </div>
      <ul>
        <li>Status right now is <strong>{status}</strong>.</li>
        <li>HRV signal is <strong>{hrv_signal}</strong>, which is a strong indicator for training tolerance.</li>
        <li>Sleep signal is <strong>{sleep_signal}</strong>, so recovery quality is materially affecting readiness.</li>
        <li>Use the manual inputs on the left to pressure-test what changes would improve tomorrow's score.</li>
      </ul>
    </div>
    """


def build_insights_table(
    day_strain: float,
    resting_heart_rate_bpm: float,
    heart_rate_variability_ms: float,
    sleep_performance: float,
    blood_oxygen: float,
    average_hr_bpm: float,
) -> pd.DataFrame:
    """Build a quick comparison table against 30-day medians."""
    metrics = [
        ("Day Strain", day_strain, "day_strain", "Lower is better today"),
        ("Resting HR", resting_heart_rate_bpm, "resting_heart_rate_bpm", "Lower is better today"),
        ("HRV", heart_rate_variability_ms, "heart_rate_variability_ms", "Higher is better today"),
        ("Sleep Performance", sleep_performance, "sleep_performance_", "Higher is better today"),
        ("Blood Oxygen", blood_oxygen, "blood_oxygen_", "Higher is better today"),
        ("Average HR", average_hr_bpm, "average_hr_bpm", "Context only"),
    ]

    rows = []
    for label, current_value, history_column, note in metrics:
        baseline = float(pd.to_numeric(HISTORY_DF[history_column], errors="coerce").median())
        delta = current_value - baseline
        rows.append(
            {
                "Signal": label,
                "Current": round(current_value, 2),
                "30D Median": round(baseline, 2),
                "Delta": round(delta, 2),
                "Interpretation": note,
            }
        )

    return pd.DataFrame(rows)


def build_raw_table() -> pd.DataFrame:
    """Return recent history for the raw data tab."""
    columns = [
        "cycle_start_time",
        "recovery_score_",
        "heart_rate_variability_ms",
        "resting_heart_rate_bpm",
        "sleep_performance_",
        "day_strain",
        "blood_oxygen_",
        "average_hr_bpm",
    ]
    available = [column for column in columns if column in HISTORY_DF.columns]
    frame = HISTORY_DF[available].sort_values("cycle_start_time", ascending=False).head(30).copy()
    frame["cycle_start_time"] = pd.to_datetime(frame["cycle_start_time"]).dt.strftime("%Y-%m-%d")
    frame = frame.rename(columns={column: DISPLAY_NAMES.get(column, column) for column in frame.columns})
    return frame.round(2)


def predict_dashboard_values(
    heart_rate_variability_ms: float,
    resting_heart_rate_bpm: float,
    day_strain: float,
    blood_oxygen: float,
    sleep_performance: float,
) -> tuple[float, str]:
    """Run the model for the manual dashboard inputs."""
    result = prediction_service.predict(
        {
            "heart_rate_variability_ms": heart_rate_variability_ms,
            "resting_heart_rate_bpm": resting_heart_rate_bpm,
            "day_strain": day_strain,
            "blood_oxygen_": blood_oxygen,
            "sleep_performance_": sleep_performance,
        }
    )
    return float(result["predicted_score"]), str(result["recommendation"])


def update_dashboard(
    day_strain: float,
    resting_heart_rate_bpm: float,
    heart_rate_variability_ms: float,
    sleep_performance: float,
    blood_oxygen: float,
    average_hr_bpm: float,
):
    """Refresh every dashboard panel from the current control values."""
    predicted_score, recommendation = predict_dashboard_values(
        heart_rate_variability_ms,
        resting_heart_rate_bpm,
        day_strain,
        blood_oxygen,
        sleep_performance,
    )
    recovery_score, contributions = calculate_recovery_score(
        heart_rate_variability_ms,
        resting_heart_rate_bpm,
        day_strain,
        blood_oxygen,
        sleep_performance,
    )

    overall_score = round((predicted_score + recovery_score) / 2, 1)
    status = classify_status(overall_score)
    hrv_signal = signal_band(
        heart_rate_variability_ms,
        HISTORY_DF["heart_rate_variability_ms"],
        ("Low", "Moderate", "High"),
    )
    sleep_signal = signal_band(
        sleep_performance,
        HISTORY_DF["sleep_performance_"],
        ("Poor", "Good", "Great"),
    )

    summary_html = build_summary_html(
        predicted_score,
        recovery_score,
        overall_score,
        status,
        hrv_signal,
        sleep_signal,
    )
    status_banner = build_status_banner(status, overall_score)
    recommendation_html = build_recommendation_html(
        status,
        overall_score,
        predicted_score,
        recovery_score,
        recommendation,
        hrv_signal,
        sleep_signal,
    )
    insights_table = build_insights_table(
        day_strain,
        resting_heart_rate_bpm,
        heart_rate_variability_ms,
        sleep_performance,
        blood_oxygen,
        average_hr_bpm,
    )

    return (
        summary_html,
        status_banner,
        make_gauge_figure("Recovery Score", recovery_score, "#35d07f"),
        make_gauge_figure("Predicted Score", predicted_score, "#4da8ff"),
        make_gauge_figure("Sleep Performance", sleep_performance, "#a366ff"),
        make_trend_figure("recovery_score_", "Recovery Score %", "#35d07f", overall_score),
        make_trend_figure("heart_rate_variability_ms", "HRV (ms)", "#4da8ff", heart_rate_variability_ms),
        make_trend_figure("resting_heart_rate_bpm", "Resting Heart Rate (bpm)", "#ff5a49", resting_heart_rate_bpm),
        make_trend_figure("day_strain", "Day Strain", "#ffb020", day_strain),
        make_trend_figure("sleep_performance_", "Sleep Performance %", "#a366ff", sleep_performance),
        make_trend_figure("blood_oxygen_", "Blood Oxygen %", "#27c8b9", blood_oxygen),
        make_feature_importance_figure(),
        make_signal_strength_figure(contributions),
        insights_table,
        recommendation_html,
        build_raw_table(),
    )


def predict_csv(file_path: str | None):
    """Run batch prediction for an uploaded CSV file."""
    if not file_path:
        return pd.DataFrame(), "Upload a CSV file to run batch predictions."

    results_df, stats = batch_service.process_csv_file(Path(file_path))
    summary = (
        f"Processed {stats['total_records']} rows. "
        f"Successful: {stats['successful']}. "
        f"Failed: {stats['failed']}. "
        f"Success rate: {stats['success_rate']}%."
    )
    return results_df, summary


def download_sample_csv():
    """Create a sample CSV for the batch workflow."""
    sample = pd.DataFrame(
        [
            {
                "heart_rate_variability_ms": 155.0,
                "resting_heart_rate_bpm": 52.0,
                "day_strain": 6.5,
                "blood_oxygen_": 98.2,
                "sleep_performance_": 86.0,
            },
            {
                "heart_rate_variability_ms": 102.0,
                "resting_heart_rate_bpm": 66.0,
                "day_strain": 18.4,
                "blood_oxygen_": 96.9,
                "sleep_performance_": 58.0,
            },
        ]
    )

    temp_file = NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    sample.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name


default_strain, default_rhr, default_hrv, default_sleep, default_spo2, default_avg_hr = latest_input_values()


with gr.Blocks(title="Health Engine Dashboard", fill_width=True) as demo:
    with gr.Row(elem_id="app-shell"):
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar-shell"):
            gr.HTML(
                """
                <div class="sidebar-title">Manual Readiness Check</div>
                <div class="sidebar-copy">
                  Enter today's values to pressure-test the model and compare them
                  with your recent physiological history.
                </div>
                """
            )

            day_strain = gr.Slider(0, 25, value=default_strain, step=0.1, label="Day Strain")
            resting_heart_rate_bpm = gr.Slider(35, 90, value=default_rhr, step=1, label="Resting HR (bpm)")
            heart_rate_variability_ms = gr.Slider(20, 220, value=default_hrv, step=1, label="HRV (ms)")
            sleep_performance = gr.Slider(0, 100, value=default_sleep, step=1, label="Sleep Performance %")
            blood_oxygen = gr.Slider(85, 100, value=default_spo2, step=0.1, label="Blood Oxygen %")
            average_hr_bpm = gr.Slider(45, 130, value=default_avg_hr, step=1, label="Average HR (bpm)")

            update_button = gr.Button("Update Dashboard", variant="primary")
            use_latest_button = gr.Button("Use Latest Recorded Values")

        with gr.Column(scale=6, elem_classes="main-shell"):
            gr.HTML(
                """
                <div class="page-title">Today's Readiness</div>
                <div class="page-copy">
                  Recovery prediction, physiological trend context, and current readiness signals.
                </div>
                """
            )

            summary_html = gr.HTML()
            status_banner = gr.HTML()

            with gr.Row():
                recovery_gauge = gr.Plot()
                predicted_gauge = gr.Plot()
                sleep_gauge = gr.Plot()

            with gr.Tabs(elem_classes="gradio-tabs"):
                with gr.Tab("Trends & History"):
                    with gr.Row():
                        recovery_trend = gr.Plot()
                        hrv_trend = gr.Plot()
                    with gr.Row():
                        rhr_trend = gr.Plot()
                        strain_trend = gr.Plot()
                    with gr.Row():
                        sleep_trend = gr.Plot()
                        spo2_trend = gr.Plot()

                with gr.Tab("Model Performance"):
                    with gr.Row():
                        feature_importance_plot = gr.Plot()
                        signal_strength_plot = gr.Plot()
                    insights_table = gr.Dataframe(
                        label="Current Signals vs 30-Day Median",
                        interactive=False,
                        wrap=True,
                    )

                with gr.Tab("Recommendations"):
                    recommendation_html = gr.HTML()

                with gr.Tab("Raw Data"):
                    raw_data_table = gr.Dataframe(
                        label="Recent Physiological History",
                        interactive=False,
                        wrap=True,
                    )

                with gr.Tab("Batch CSV"):
                    with gr.Row():
                        csv_input = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
                        sample_file = gr.File(label="Sample CSV", interactive=False)
                    with gr.Row():
                        sample_button = gr.Button("Create Sample CSV")
                        batch_button = gr.Button("Run Batch Prediction", variant="primary")
                    batch_summary = gr.Textbox(label="Batch Summary", interactive=False)
                    batch_output = gr.Dataframe(label="Predictions", wrap=True, interactive=False)

                    sample_button.click(download_sample_csv, outputs=sample_file)
                    batch_button.click(predict_csv, inputs=csv_input, outputs=[batch_output, batch_summary])

    dashboard_inputs = [
        day_strain,
        resting_heart_rate_bpm,
        heart_rate_variability_ms,
        sleep_performance,
        blood_oxygen,
        average_hr_bpm,
    ]
    dashboard_outputs = [
        summary_html,
        status_banner,
        recovery_gauge,
        predicted_gauge,
        sleep_gauge,
        recovery_trend,
        hrv_trend,
        rhr_trend,
        strain_trend,
        sleep_trend,
        spo2_trend,
        feature_importance_plot,
        signal_strength_plot,
        insights_table,
        recommendation_html,
        raw_data_table,
    ]

    update_button.click(update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)

    use_latest_button.click(
        latest_input_values,
        outputs=dashboard_inputs,
    ).then(
        update_dashboard,
        inputs=dashboard_inputs,
        outputs=dashboard_outputs,
    )

    demo.load(update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)


if __name__ == "__main__":
    _, local_url, share_url = demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=find_server_port(),
        share=env_flag("GRADIO_SHARE"),
        prevent_thread_lock=True,
        css=CUSTOM_CSS,
    )
    print(f"LOCAL_URL={local_url}", flush=True)
    if share_url:
        print(f"PUBLIC_URL={share_url}", flush=True)
    demo.block_thread()
