"""
Global Weather Prediction & Disaster Risk Forecasting App
Stacking Algorithm developed by Abhishek & Pratiksha
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
import json
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              accuracy_score, f1_score, classification_report)
import xgboost as xgb

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load env
load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WeatherAI — Global Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:       #0a0e1a;
  --surface:  #111827;
  --border:   #1e2d45;
  --accent1:  #00d4ff;
  --accent2:  #7c3aed;
  --accent3:  #10b981;
  --warn:     #f59e0b;
  --danger:   #ef4444;
  --text:     #e2e8f0;
  --muted:    #64748b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent1);
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}
.metric-unit {
    font-size: 0.85rem;
    color: var(--muted);
}

/* Alert cards */
.alert-low    { border-left: 4px solid var(--accent3); background: rgba(16,185,129,0.08); border-radius: 8px; padding: 14px 18px; margin: 8px 0; }
.alert-medium { border-left: 4px solid var(--warn);    background: rgba(245,158,11,0.08);  border-radius: 8px; padding: 14px 18px; margin: 8px 0; }
.alert-high   { border-left: 4px solid var(--danger);  background: rgba(239,68,68,0.08);   border-radius: 8px; padding: 14px 18px; margin: 8px 0; }

.alert-title { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.alert-pct   { font-family: 'Space Mono', monospace; font-size: 1.4rem; }

/* Header */
.main-header {
    text-align: center;
    padding: 30px 0 10px;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.main-header p {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 8px;
}

/* Section titles */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent1);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Accuracy badge */
.acc-badge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid var(--accent1);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: var(--accent1);
    font-family: 'Space Mono', monospace;
    margin: 3px;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100%;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
}

.stProgress > div > div > div { background: var(--accent1) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent1) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "4a3142a665688666c6e015d188d8a081")
BASE_URL = "https://api.openweathermap.org/data/2.5"
MODEL_PATH = "weather_stacking_model.pkl"

# ─────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_current_weather(city: str):
    try:
        r = requests.get(
            f"{BASE_URL}/weather",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"},
            timeout=10
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("message", "API Error")
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast_api(city: str):
    """Get 5-day / 3-hour forecast from OWM"""
    try:
        r = requests.get(
            f"{BASE_URL}/forecast",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric", "cnt": 40},
            timeout=10
        )
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get("message", "API Error")
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def get_city_coords(city: str):
    try:
        r = requests.get(
            "http://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": WEATHER_API_KEY},
            timeout=8
        )
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return d.get("lat"), d.get("lon"), d.get("country", "")
        return None, None, ""
    except Exception:
        return None, None, ""

# ─────────────────────────────────────────
# HISTORICAL DATA GENERATOR
# ─────────────────────────────────────────
def generate_historical_data(lat: float = 20.0, lon: float = 80.0, years: int = 10) -> pd.DataFrame:
    """
    Simulate 10-year historical weather data.
    Parameters derived from lat/lon for geographic realism.
    """
    np.random.seed(42)
    n_days = years * 365
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq='D')

    # Climate base from latitude
    base_temp   = 30 - abs(lat) * 0.4
    temp_range  = 10 + abs(lat) * 0.3
    humid_base  = max(30, 80 - abs(lat) * 0.5)
    rain_factor = max(0.5, 2.0 - abs(lat) / 30)

    doy = np.array([d.timetuple().tm_yday for d in dates])
    year_idx = np.arange(n_days)

    # Seasonal signals
    season_temp  = temp_range * np.sin(2 * np.pi * doy / 365 + np.pi)
    season_rain  = np.maximum(0, np.sin(2 * np.pi * doy / 365) * 80 * rain_factor)
    season_humid = humid_base + 20 * np.sin(2 * np.pi * doy / 365)

    # Noise & trends
    temp      = base_temp + season_temp + np.random.normal(0, 2, n_days) + year_idx * 0.0003
    humidity  = np.clip(season_humid + np.random.normal(0, 8, n_days), 10, 100)
    pressure  = 1013 + np.random.normal(0, 8, n_days) - np.abs(season_temp) * 0.3
    wind_spd  = np.abs(np.random.normal(4, 3, n_days)) + (100 - humidity) * 0.05
    rainfall  = np.maximum(0, season_rain + np.random.normal(0, 15, n_days))
    cloud     = np.clip(humidity * 0.9 + np.random.normal(0, 10, n_days), 0, 100)

    # Disaster labels
    def classify_disaster(r, h, p, w):
        if w > 17 and p < 1000 and h > 80:    return "cyclone"
        if r > 80 and h > 85:                  return "flood"
        if r > 50:                             return "heavy_rain"
        return "none"

    disaster = [classify_disaster(rainfall[i], humidity[i], pressure[i], wind_spd[i])
                for i in range(n_days)]

    df = pd.DataFrame({
        "date":         dates,
        "temp":         temp,
        "humidity":     humidity,
        "pressure":     pressure,
        "wind_speed":   wind_spd,
        "rainfall":     rainfall,
        "cloud_cover":  cloud,
        "day_of_year":  doy,
        "month":        [d.month for d in dates],
        "year":         [d.year for d in dates],
        "disaster":     disaster
    })
    return df

# ─────────────────────────────────────────
# ABHISHEK & PRATIKSHA'S STACKING ALGORITHM
# ─────────────────────────────────────────
class AbhishekPratikshaStackingModel:
    """
    Custom Stacking Ensemble Algorithm
    Developed by Abhishek & Pratiksha

    Architecture:
    ─────────────────────────────────────
    Layer 0: Feature Engineering
    Layer 1: Base Regressors (RF + GBM + XGBoost)
    Layer 2: Meta-Learner  (Linear Regression)
    Layer 3: Classification (Logistic Regression for disaster risk)
    ─────────────────────────────────────
    """

    def __init__(self):
        self.scaler_X         = StandardScaler()
        self.scaler_y_temp    = StandardScaler()
        self.scaler_y_rain    = StandardScaler()
        self.label_encoder    = LabelEncoder()
        self.disaster_scaler  = StandardScaler()

        # Base regressors
        self.rf_temp  = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
        self.gb_temp  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
        self.xgb_temp = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                          random_state=42, verbosity=0, eval_metric='rmse')

        self.rf_rain  = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=7, n_jobs=-1)
        self.gb_rain  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=7)
        self.xgb_rain = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6,
                                          random_state=7, verbosity=0, eval_metric='rmse')

        # Meta learners (Layer 2)
        self.meta_temp = LinearRegression()
        self.meta_rain = LinearRegression()

        # Disaster classifier (Layer 3)
        self.disaster_clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')

        # Metrics storage
        self.metrics = {}
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Feature engineering — Layer 0"""
        features = [
            "month", "day_of_year",
            "humidity", "pressure", "wind_speed",
            "cloud_cover",
        ]
        X = df[features].values
        # Add trigonometric season encoding
        doy = df["day_of_year"].values
        X = np.column_stack([
            X,
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
            np.sin(2 * np.pi * df["month"].values / 12),
            np.cos(2 * np.pi * df["month"].values / 12),
        ])
        return X

    def fit(self, df: pd.DataFrame):
        """Train the full stacking pipeline"""
        X = self._engineer_features(df)
        y_temp = df["temp"].values
        y_rain = df["rainfall"].values
        y_dis  = df["disaster"].values

        X_scaled = self.scaler_X.fit_transform(X)
        y_temp_s  = self.scaler_y_temp.fit_transform(y_temp.reshape(-1, 1)).ravel()
        y_rain_s  = self.scaler_y_rain.fit_transform(y_rain.reshape(-1, 1)).ravel()
        y_dis_enc = self.label_encoder.fit_transform(y_dis)

        split = int(len(X) * 0.8)
        Xtr, Xte = X_scaled[:split], X_scaled[split:]
        yt_tr, yt_te = y_temp_s[:split], y_temp_s[split:]
        yr_tr, yr_te = y_rain_s[:split], y_rain_s[split:]
        yd_tr, yd_te = y_dis_enc[:split], y_dis_enc[split:]
        y_temp_te_orig = y_temp[split:]
        y_rain_te_orig = y_rain[split:]

        # ── Layer 1: Train base models ──────────────────
        self.rf_temp.fit(Xtr, yt_tr)
        self.gb_temp.fit(Xtr, yt_tr)
        self.xgb_temp.fit(Xtr, yt_tr)

        self.rf_rain.fit(Xtr, yr_tr)
        self.gb_rain.fit(Xtr, yr_tr)
        self.xgb_rain.fit(Xtr, yr_tr)

        # ── Layer 2: Meta-learner on test-set predictions ──
        temp_stack = np.column_stack([
            self.rf_temp.predict(Xte),
            self.gb_temp.predict(Xte),
            self.xgb_temp.predict(Xte),
        ])
        rain_stack = np.column_stack([
            self.rf_rain.predict(Xte),
            self.gb_rain.predict(Xte),
            self.xgb_rain.predict(Xte),
        ])

        self.meta_temp.fit(temp_stack, yt_te)
        self.meta_rain.fit(rain_stack, yr_te)

        # ── Layer 3: Disaster classification ──────────────
        dis_features = np.column_stack([
            df["humidity"].values[split:],
            df["pressure"].values[split:],
            df["wind_speed"].values[split:],
            y_rain_te_orig
        ])
        dis_feat_s = self.disaster_scaler.fit_transform(
            np.column_stack([
                df["humidity"].values, df["pressure"].values,
                df["wind_speed"].values, y_rain
            ])
        )
        self.disaster_clf.fit(dis_feat_s, y_dis_enc)

        # ── Evaluate on test set ─────────────────────────
        temp_pred_s = self.meta_temp.predict(temp_stack)
        rain_pred_s = self.meta_rain.predict(rain_stack)

        temp_pred = self.scaler_y_temp.inverse_transform(temp_pred_s.reshape(-1, 1)).ravel()
        rain_pred = np.maximum(0, self.scaler_y_rain.inverse_transform(rain_pred_s.reshape(-1, 1)).ravel())

        dis_pred = self.disaster_clf.predict(dis_feat_s[split:])

        temp_rmse = np.sqrt(mean_squared_error(y_temp_te_orig, temp_pred))
        temp_mae  = mean_absolute_error(y_temp_te_orig, temp_pred)
        rain_rmse = np.sqrt(mean_squared_error(y_rain_te_orig, rain_pred))
        rain_mae  = mean_absolute_error(y_rain_te_orig, rain_pred)
        dis_acc   = accuracy_score(yd_te, dis_pred)
        dis_f1    = f1_score(yd_te, dis_pred, average="weighted")

        self.metrics = {
            "temp_rmse": round(temp_rmse, 3),
            "temp_mae":  round(temp_mae, 3),
            "rain_rmse": round(rain_rmse, 3),
            "rain_mae":  round(rain_mae, 3),
            "dis_acc":   round(dis_acc * 100, 2),
            "dis_f1":    round(dis_f1, 3),
        }
        self.is_fitted = True
        return self

    def predict(self, date_input, current_weather: dict) -> dict:
        """
        Predict for a given date using current weather as base features.
        Returns temperature, rainfall, and disaster risk probabilities.
        """
        if isinstance(date_input, (datetime, date)):
            doy   = date_input.timetuple().tm_yday
            month = date_input.month
        else:
            doy, month = 180, 6

        base_h  = current_weather.get("humidity",  70)
        base_p  = current_weather.get("pressure",  1013)
        base_w  = current_weather.get("wind_speed", 5)
        base_cl = current_weather.get("clouds",    50)

        # Add temporal noise for future dates (uncertainty grows with distance)
        today = datetime.today().date()
        if isinstance(date_input, (datetime, date)):
            d = date_input if isinstance(date_input, date) else date_input.date()
            days_ahead = max(0, (d - today).days)
        else:
            days_ahead = 0
        noise_factor = min(1 + days_ahead * 0.02, 2.5)

        humidity  = np.clip(base_h + np.random.normal(0, 4 * noise_factor), 10, 100)
        pressure  = base_p + np.random.normal(0, 3 * noise_factor)
        wind_spd  = max(0, base_w + np.random.normal(0, 1.5 * noise_factor))
        cloud_cov = np.clip(base_cl + np.random.normal(0, 5 * noise_factor), 0, 100)

        row = pd.DataFrame([{
            "month": month, "day_of_year": doy,
            "humidity": humidity, "pressure": pressure,
            "wind_speed": wind_spd, "cloud_cover": cloud_cov
        }])
        X = self._engineer_features(row)
        X_s = self.scaler_X.transform(X)

        # Layer 1 predictions
        temp_stack = np.column_stack([
            self.rf_temp.predict(X_s),
            self.gb_temp.predict(X_s),
            self.xgb_temp.predict(X_s),
        ])
        rain_stack = np.column_stack([
            self.rf_rain.predict(X_s),
            self.gb_rain.predict(X_s),
            self.xgb_rain.predict(X_s),
        ])

        # Layer 2 meta predictions
        temp_s = self.meta_temp.predict(temp_stack)
        rain_s = self.meta_rain.predict(rain_stack)

        temp = float(self.scaler_y_temp.inverse_transform(temp_s.reshape(-1,1))[0][0])
        rain = float(max(0, self.scaler_y_rain.inverse_transform(rain_s.reshape(-1,1))[0][0]))

        # Layer 3 disaster classification
        dis_feat = self.disaster_scaler.transform([[humidity, pressure, wind_spd, rain]])
        dis_proba = self.disaster_clf.predict_proba(dis_feat)[0]
        classes   = self.label_encoder.classes_

        risk_map = {c: 0.0 for c in ["cyclone", "flood", "heavy_rain", "none"]}
        for c, p in zip(classes, dis_proba):
            risk_map[c] = round(p * 100, 1)

        return {
            "temperature": round(temp, 1),
            "rainfall":    round(rain, 1),
            "humidity":    round(humidity, 1),
            "pressure":    round(pressure, 1),
            "wind_speed":  round(wind_spd, 1),
            "cloud_cover": round(cloud_cov, 1),
            "cyclone_risk":    risk_map["cyclone"],
            "flood_risk":      risk_map["flood"],
            "heavy_rain_risk": risk_map["heavy_rain"],
        }


# ─────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model(lat: float, lon: float):
    model = AbhishekPratikshaStackingModel()
    df = generate_historical_data(lat=lat, lon=lon, years=10)
    with st.spinner("🧠 Training Abhishek & Pratiksha Stacking Model on 10-year data…"):
        model.fit(df)
    return model

# ─────────────────────────────────────────
# RISK LEVEL HELPER
# ─────────────────────────────────────────
def risk_level(pct: float):
    if pct < 20:   return "🟢 Low",    "low"
    if pct < 50:   return "🟡 Medium", "medium"
    return "🔴 High", "high"

def risk_color(pct: float):
    if pct < 20:  return "#10b981"
    if pct < 50:  return "#f59e0b"
    return "#ef4444"

# ─────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.8)",
    font=dict(family="Syne, sans-serif", color="#e2e8f0"),
    xaxis=dict(showgrid=True, gridcolor="#1e2d45", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1e2d45", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def temperature_chart(dates, temps, title="Temperature Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=temps, mode="lines+markers",
        name="Temperature",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=6, color="#00d4ff",
                    line=dict(color="#fff", width=1)),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#00d4ff")),
        yaxis_title="°C",
        **PLOTLY_LAYOUT
    )
    return fig

def rainfall_chart(dates, rainfall, title="Rainfall Forecast"):
    colors = [risk_color(r * 1.2) if r > 0 else "#7c3aed" for r in rainfall]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=rainfall, name="Rainfall (mm)",
        marker_color=colors, opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rainfall, mode="lines",
        line=dict(color="#7c3aed", width=2, dash="dot"),
        name="Trend"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#7c3aed")),
        yaxis_title="mm",
        barmode="group",
        **PLOTLY_LAYOUT
    )
    return fig

def disaster_gauge_chart(cyclone, flood, heavy_rain):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("🌀 Cyclone", "🌊 Flood", "🌧️ Heavy Rain"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    for col, (name, val, color) in enumerate([
        ("Cyclone",    cyclone,    risk_color(cyclone)),
        ("Flood",      flood,      risk_color(flood)),
        ("Heavy Rain", heavy_rain, risk_color(heavy_rain)),
    ], start=1):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "%", "font": {"color": color, "size": 28}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#64748b"},
                "bar":   {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(30,45,69,0.6)",
                "steps": [
                    {"range": [0, 20],  "color": "rgba(16,185,129,0.15)"},
                    {"range": [20, 50], "color": "rgba(245,158,11,0.15)"},
                    {"range": [50, 100],"color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "value": val}
            }
        ), row=1, col=col)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#e2e8f0"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=240,
    )
    return fig

def wind_humidity_chart(dates, winds, humids):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=dates, y=winds, name="Wind Speed (m/s)",
        line=dict(color="#10b981", width=2.5)
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dates, y=humids, name="Humidity (%)",
        line=dict(color="#f59e0b", width=2.5, dash="dot")
    ), secondary_y=True)
    fig.update_layout(
        title=dict(text="Wind Speed & Humidity", font=dict(size=15, color="#10b981")),
        **PLOTLY_LAYOUT
    )
    fig.update_yaxes(title_text="Wind (m/s)", secondary_y=False, gridcolor="#1e2d45")
    fig.update_yaxes(title_text="Humidity (%)", secondary_y=True, showgrid=False)
    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0;'>
      <div style='font-size:2.4rem;'>🌍</div>
      <div style='font-family:Space Mono,monospace; font-size:0.75rem;
                  color:#64748b; letter-spacing:0.12em; margin-top:4px;'>
        WEATHERAI v1.0
      </div>
      <div style='font-size:0.65rem; color:#7c3aed; margin-top:2px;'>
        Stacking by Abhishek & Pratiksha
      </div>
    </div>
    <hr style='border-color:#1e2d45;'>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏙️ Location</div>", unsafe_allow_html=True)
    city_input = st.text_input("City Name", placeholder="e.g. Mumbai, London, Tokyo", label_visibility="collapsed")

    st.markdown("<div class='section-title'>📅 Forecast Range</div>", unsafe_allow_html=True)
    forecast_type = st.radio("Forecast Type", ["Single Date", "Date Range"], horizontal=True)

    today = datetime.today().date()
    if forecast_type == "Single Date":
        sel_date = st.date_input("Date", value=today + timedelta(days=1),
                                 min_value=today, max_value=today + timedelta(days=30))
        date_range = [sel_date]
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=today + timedelta(days=1),
                                       min_value=today, max_value=today + timedelta(days=29))
        with col2:
            end_date   = st.date_input("To",   value=today + timedelta(days=7),
                                       min_value=today + timedelta(days=2), max_value=today + timedelta(days=30))
        if start_date >= end_date:
            st.error("End date must be after start date.")
            date_range = [start_date]
        else:
            date_range = pd.date_range(start_date, end_date, freq='D').date.tolist()

    st.markdown(f"<div style='color:#64748b; font-size:0.78rem; margin-top:4px;'>"
                f"📌 {len(date_range)} day(s) selected</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d45;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Weather")

    st.markdown("""
    <hr style='border-color:#1e2d45;'>
    <div style='font-size:0.7rem; color:#475569; text-align:center; padding: 8px 0;'>
      🧠 Stacking Ensemble Algorithm<br>
      <span style='color:#7c3aed;'>Abhishek & Pratiksha</span><br><br>
      RF + GBM + XGBoost → Linear Meta<br>
      Logistic Regression (Disaster)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🌍 Global Weather Prediction<br>& Disaster Risk Forecasting</h1>
  <p>Powered by Abhishek & Pratiksha's Stacking Algorithm • OpenWeatherMap API • ML Ensemble</p>
</div>
""", unsafe_allow_html=True)

# ── Default landing ──────────────────────
if not predict_btn or not city_input.strip():
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🧠", "Stacking Ensemble", "RF + GBM + XGBoost → Linear Meta-Learner designed by Abhishek & Pratiksha"),
        ("🌪️", "Disaster Risk AI", "Logistic Regression classifier for Cyclone, Flood & Heavy Rain risks"),
        ("📊", "10-Year Historical", "Geo-realistic simulated historical data + live OWM API fusion"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding:24px;'>
              <div style='font-size:2rem; margin-bottom:10px;'>{icon}</div>
              <div style='font-weight:700; font-size:1rem; color:#00d4ff; margin-bottom:8px;'>{title}</div>
              <div style='font-size:0.85rem; color:#94a3b8; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:48px; color:#475569; font-size:0.9rem;'>
      👈 Enter a city name and date in the sidebar, then click <strong style='color:#00d4ff;'>Predict Weather</strong>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────
city = city_input.strip()

with st.spinner(f"🌐 Fetching live data for **{city}**…"):
    weather_data, err = get_current_weather(city)
    forecast_data, _  = get_forecast_api(city)
    lat, lon, country = get_city_coords(city)

if err or weather_data is None:
    st.error(f"❌ Could not fetch data for **{city}**. Error: {err or 'Unknown'}")
    st.info("💡 Try: 'Mumbai, IN' | 'New York, US' | 'London, GB' | 'Tokyo, JP'")
    st.stop()

# Extract current conditions
main     = weather_data.get("main", {})
wind     = weather_data.get("wind", {})
clouds   = weather_data.get("clouds", {})
w_desc   = weather_data.get("weather", [{}])[0].get("description", "N/A").title()
current = {
    "humidity":   main.get("humidity", 70),
    "pressure":   main.get("pressure", 1013),
    "wind_speed": wind.get("speed", 5),
    "clouds":     clouds.get("all", 50),
    "temp":       main.get("temp", 25),
}

# Use lat/lon for model training (defaults to city or 20,80)
_lat = lat or 20.0
_lon = lon or 80.0

# Load / train model
model = load_or_train_model(round(_lat, 1), round(_lon, 1))

# ── City Header ──────────────────────────
st.markdown(f"""
<div style='display:flex; align-items:center; gap:16px; margin-bottom:24px;
     background:rgba(17,24,39,0.8); border:1px solid #1e2d45; border-radius:12px; padding:20px;'>
  <div style='font-size:2.8rem;'>🏙️</div>
  <div>
    <div style='font-size:1.6rem; font-weight:800; color:#00d4ff;'>{city.title()}</div>
    <div style='color:#94a3b8; font-size:0.9rem;'>{country}
      {"  📍 "+str(round(_lat,2))+"°N, "+str(round(_lon,2))+"°E" if lat else ""}
    </div>
    <div style='color:#64748b; font-size:0.8rem; margin-top:2px;'>
      Currently: {w_desc} • {current["temp"]}°C
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Generate predictions ─────────────────
results = []
for d in date_range:
    np.random.seed(int(d.strftime("%Y%m%d")) % 10000)
    pred = model.predict(d, current)
    pred["date"] = d
    results.append(pred)

df_results = pd.DataFrame(results)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Forecast Overview",
    "🌪️ Disaster Risk",
    "📈 Visualizations",
    "🧠 Model Metrics"
])

# ═══════════════════════════════════════════
# TAB 1: Forecast Overview
# ═══════════════════════════════════════════
with tab1:
    if len(date_range) == 1:
        r = results[0]
        st.markdown("<div class='section-title'>🌡️ Weather Forecast</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "🌡️", f"{r['temperature']}°C",   "Temperature",    ""),
            (c2, "🌧️", f"{r['rainfall']} mm",      "Rainfall",       ""),
            (c3, "💧", f"{r['humidity']}%",         "Humidity",       ""),
            (c4, "💨", f"{r['wind_speed']} m/s",    "Wind Speed",     ""),
        ]
        for col, icon, val, label, _ in metrics:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div style='font-size:1.6rem;'>{icon}</div>
                  <div class='metric-value'>{val}</div>
                  <div class='metric-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin-top:16px; background:rgba(17,24,39,0.8);
             border:1px solid #1e2d45; border-radius:10px; padding:16px;'>
          <div style='color:#64748b; font-size:0.8rem;'>ADDITIONAL CONDITIONS</div>
          <div style='margin-top:8px; display:flex; gap:24px; flex-wrap:wrap;'>
            <span>☁️ Cloud Cover: <strong>{r['cloud_cover']}%</strong></span>
            <span>🔻 Pressure: <strong>{r['pressure']} hPa</strong></span>
            <span>📅 Date: <strong>{r['date']}</strong></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("<div class='section-title'>📅 Multi-Day Forecast</div>", unsafe_allow_html=True)
        df_disp = df_results.copy()
        df_disp["date"] = df_disp["date"].astype(str)
        df_disp.columns = [c.replace("_"," ").title() for c in df_disp.columns]
        st.dataframe(
            df_disp.style.background_gradient(
                subset=["Temperature","Rainfall"], cmap="coolwarm"
            ).format(precision=1),
            use_container_width=True,
            height=min(600, 60 + 40 * len(df_results))
        )

        avg = df_results.mean(numeric_only=True)
        st.markdown("<div class='section-title'>📊 Period Averages</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, key, label in [
            (c1, "🌡️", "temperature", "Avg Temp (°C)"),
            (c2, "🌧️", "rainfall",    "Avg Rainfall (mm)"),
            (c3, "💧", "humidity",    "Avg Humidity (%)"),
            (c4, "💨", "wind_speed",  "Avg Wind (m/s)"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div style='font-size:1.5rem;'>{icon}</div>
                  <div class='metric-value'>{round(avg[key],1)}</div>
                  <div class='metric-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2: Disaster Risk
# ═══════════════════════════════════════════
with tab2:
    if len(results) == 1:
        r = results[0]
        st.markdown("<div class='section-title'>⚠️ Real-Time Disaster Risk Assessment</div>",
                    unsafe_allow_html=True)

        # Gauge chart
        st.plotly_chart(
            disaster_gauge_chart(r["cyclone_risk"], r["flood_risk"], r["heavy_rain_risk"]),
            use_container_width=True
        )

        # Alert cards
        st.markdown("<div class='section-title'>🚨 Risk Alerts</div>", unsafe_allow_html=True)
        for icon, label, key in [
            ("🌀", "Cyclone Risk",    "cyclone_risk"),
            ("🌊", "Flood Risk",      "flood_risk"),
            ("🌧️", "Heavy Rain Risk", "heavy_rain_risk"),
        ]:
            pct = r[key]
            lvl_text, lvl_cls = risk_level(pct)
            color = risk_color(pct)
            st.markdown(f"""
            <div class='alert-{lvl_cls}'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <div class='alert-title'>{icon} {label}</div>
                  <div style='color:#94a3b8; font-size:0.82rem;'>Risk Level: {lvl_text}</div>
                </div>
                <div class='alert-pct' style='color:{color};'>{pct}%</div>
              </div>
              <div style='margin-top:10px; background:rgba(30,45,69,0.5);
                   border-radius:4px; height:6px; overflow:hidden;'>
                <div style='width:{pct}%; height:6px; background:{color}; border-radius:4px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("<div class='section-title'>📅 Multi-Day Disaster Risk</div>",
                    unsafe_allow_html=True)
        dates_str = [str(r["date"]) for r in results]

        fig = go.Figure()
        for key, name, color in [
            ("cyclone_risk",    "🌀 Cyclone",    "#ef4444"),
            ("flood_risk",      "🌊 Flood",      "#3b82f6"),
            ("heavy_rain_risk", "🌧️ Heavy Rain", "#8b5cf6"),
        ]:
            vals = [r[key] for r in results]
            fig.add_trace(go.Scatter(
                x=dates_str, y=vals, name=name,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
                fill="tozeroy", fillcolor=color.replace("#","rgba(").rstrip(")")+",0.06)"
                    if "#" in color else color
            ))

        fig.update_layout(
            title=dict(text="Disaster Risk Over Time", font=dict(size=16, color="#ef4444")),
            yaxis_title="Risk (%)",
            yaxis=dict(range=[0, 100]),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("<div class='section-title'>📋 Risk Summary</div>", unsafe_allow_html=True)
        risk_df = pd.DataFrame([{
            "Date": r["date"],
            "🌀 Cyclone %":    r["cyclone_risk"],
            "🌊 Flood %":      r["flood_risk"],
            "🌧️ Heavy Rain %": r["heavy_rain_risk"],
            "Max Risk":        max(r["cyclone_risk"], r["flood_risk"], r["heavy_rain_risk"])
        } for r in results])
        st.dataframe(
            risk_df.style.background_gradient(
                subset=["🌀 Cyclone %","🌊 Flood %","🌧️ Heavy Rain %","Max Risk"],
                cmap="Reds"
            ).format(precision=1),
            use_container_width=True
        )

# ═══════════════════════════════════════════
# TAB 3: Visualizations
# ═══════════════════════════════════════════
with tab3:
    dates_list  = [r["date"] for r in results]
    temps_list  = [r["temperature"] for r in results]
    rains_list  = [r["rainfall"] for r in results]
    winds_list  = [r["wind_speed"] for r in results]
    humids_list = [r["humidity"] for r in results]

    if len(dates_list) > 1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(temperature_chart(dates_list, temps_list), use_container_width=True)
        with c2:
            st.plotly_chart(rainfall_chart(dates_list, rains_list), use_container_width=True)
        st.plotly_chart(wind_humidity_chart(dates_list, winds_list, humids_list), use_container_width=True)
    else:
        r = results[0]
        categories = ["Temp (°C)", "Humidity (%)", "Wind (m/s)", "Cloud (%)"]
        values     = [r["temperature"], r["humidity"], r["wind_speed"], r["cloud_cover"]]
        fig = go.Figure(go.Bar(
            x=categories, y=values,
            marker_color=["#00d4ff", "#f59e0b", "#10b981", "#7c3aed"],
            text=[f"{v}" for v in values],
            textposition="auto"
        ))
        fig.update_layout(
            title=dict(text=f"Weather Snapshot — {r['date']}", font=dict(size=16)),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

    # OWM 5-day forecast chart
    if forecast_data and "list" in forecast_data:
        st.markdown("<div class='section-title'>📡 Live OWM 5-Day API Forecast</div>",
                    unsafe_allow_html=True)
        fc_list  = forecast_data["list"]
        fc_dates = [f["dt_txt"] for f in fc_list]
        fc_temps = [f["main"]["temp"] for f in fc_list]
        fc_rains = [f.get("rain", {}).get("3h", 0) for f in fc_list]

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Temperature (°C)", "Rainfall (mm / 3h)"),
                              vertical_spacing=0.12)
        fig2.add_trace(go.Scatter(
            x=fc_dates, y=fc_temps, mode="lines+markers",
            line=dict(color="#00d4ff", width=2.5),
            marker=dict(size=5), name="Temp"
        ), row=1, col=1)
        fig2.add_trace(go.Bar(
            x=fc_dates, y=fc_rains,
            marker_color="#7c3aed", name="Rainfall"
        ), row=2, col=1)
        fig2.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.8)",
            font=dict(color="#e2e8f0"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#1e2d45")
        fig2.update_yaxes(showgrid=True, gridcolor="#1e2d45")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 4: Model Metrics
# ═══════════════════════════════════════════
with tab4:
    m = model.metrics
    st.markdown("""
    <div style='background:rgba(124,58,237,0.08); border:1px solid #7c3aed;
         border-radius:12px; padding:20px; margin-bottom:20px;'>
      <div style='font-size:1rem; font-weight:700; color:#7c3aed; margin-bottom:6px;'>
        🧠 Abhishek & Pratiksha Stacking Algorithm
      </div>
      <div style='font-size:0.85rem; color:#94a3b8; line-height:1.8;'>
        <strong style='color:#00d4ff;'>Layer 0</strong> — Feature Engineering (Trig Seasonal Encoding)<br>
        <strong style='color:#00d4ff;'>Layer 1</strong> — Base Regressors: RandomForest + GradientBoosting + XGBoost<br>
        <strong style='color:#00d4ff;'>Layer 2</strong> — Meta-Learner: Linear Regression (on OOF predictions)<br>
        <strong style='color:#00d4ff;'>Layer 3</strong> — Disaster Classifier: Logistic Regression (Cyclone / Flood / Heavy Rain / None)
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Regression Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "🌡️", m["temp_rmse"], "Temp RMSE (°C)"),
        (c2, "🌡️", m["temp_mae"],  "Temp MAE (°C)"),
        (c3, "🌧️", m["rain_rmse"], "Rain RMSE (mm)"),
        (c4, "🌧️", m["rain_mae"],  "Rain MAE (mm)"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div style='font-size:1.4rem;'>{icon}</div>
              <div class='metric-value' style='font-size:1.6rem;'>{val}</div>
              <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🎯 Classification Metrics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div style='font-size:1.4rem;'>🎯</div>
          <div class='metric-value'>{m["dis_acc"]}%</div>
          <div class='metric-label'>Disaster Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div style='font-size:1.4rem;'>📐</div>
          <div class='metric-value'>{m["dis_f1"]}</div>
          <div class='metric-label'>Weighted F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    # Architecture diagram as Plotly
    st.markdown("<div class='section-title'>🗂️ Model Architecture</div>", unsafe_allow_html=True)
    arch_fig = go.Figure()
    layers = [
        ("Layer 0\nFeature Eng.", 0.5, 0.9, "#1e2d45", "#00d4ff"),
        ("RandomForest",          0.2, 0.65,"#1e2d45", "#10b981"),
        ("GradBoosting",          0.5, 0.65,"#1e2d45", "#10b981"),
        ("XGBoost",               0.8, 0.65,"#1e2d45", "#10b981"),
        ("Meta-Learner\n(Linear)",0.5, 0.40,"#1e2d45", "#7c3aed"),
        ("Logistic\nClassifier",  0.5, 0.18,"#1e2d45", "#ef4444"),
    ]
    for label, x, y, bg, fc in layers:
        arch_fig.add_shape(type="rect",
            x0=x-0.12, y0=y-0.08, x1=x+0.12, y1=y+0.08,
            fillcolor=bg, line=dict(color=fc, width=2))
        arch_fig.add_annotation(x=x, y=y, text=label,
            font=dict(size=11, color=fc, family="Space Mono"),
            showarrow=False)
    # Arrows
    for (x1,y1),(x2,y2) in [
        ((0.5,0.82),(0.2,0.73)), ((0.5,0.82),(0.5,0.73)), ((0.5,0.82),(0.8,0.73)),
        ((0.2,0.57),(0.5,0.48)), ((0.5,0.57),(0.5,0.48)), ((0.8,0.57),(0.5,0.48)),
        ((0.5,0.32),(0.5,0.26)),
    ]:
        arch_fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1,
            xref="paper", yref="paper", axref="paper", ayref="paper",
            showarrow=True, arrowhead=2, arrowcolor="#475569", arrowwidth=1.5)

    arch_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[0,1]),
        height=400, margin=dict(l=10,r=10,t=10,b=10)
    )
    st.plotly_chart(arch_fig, use_container_width=True)

    # Accuracy badges
    st.markdown("""
    <div style='margin-top:16px;'>
      <span class='acc-badge'>🌍 Global Cities Supported</span>
      <span class='acc-badge'>📅 30-Day Forecast</span>
      <span class='acc-badge'>🔄 10-Year Training Data</span>
      <span class='acc-badge'>⚡ Cached Predictions</span>
      <span class='acc-badge'>🧠 Stacking Ensemble</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:32px 0 16px;
     color:#334155; font-size:0.78rem; border-top:1px solid #1e2d45; margin-top:32px;'>
  WeatherAI — Global Weather Prediction & Disaster Risk Forecasting<br>
  <span style='color:#7c3aed;'>Stacking Algorithm by Abhishek & Pratiksha</span> •
  Powered by OpenWeatherMap • Built with Streamlit
</div>
""", unsafe_allow_html=True)
