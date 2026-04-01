
"""
AtmosAI — Weather Prediction Engine (Backend)
BTech Final Year Project

Stacking Ensemble Model for Weather Prediction
- Base Learners: Random Forest, XGBoost, Gradient Boosting, LSTM-style SVR, Ridge Regression
- Meta Learner: Logistic/Linear Regression on OOF predictions
- Dataset: Synthetic 100-year historical data (replace with NOAA/ERA5 for production)

Author: [Your Name]
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings, json, math, random, os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  1. SYNTHETIC 100-YEAR DATASET GENERATOR
#     Replace this with: pd.read_csv("noaa_weather.csv")
# ─────────────────────────────────────────────
def generate_historical_dataset(n_years=100, city_lat=40.7, city_lon=-74.0):
    """
    Simulates 100 years of daily weather data using physics-inspired formulas.
    Features: day_of_year, month, year, lat, lon, lag features, rolling stats.
    Target: next_day_temp (regression) + rain_class (classification)
    """
    np.random.seed(42)
    n_days = n_years * 365
    dates = pd.date_range("1924-01-01", periods=n_days, freq="D")

    doy = dates.dayofyear.values  # day of year 1–365
    year = dates.year.values
    month = dates.month.values

    # Seasonal temperature (sinusoidal) + long-term climate drift (warming trend)
    base_temp = (
        15
        + 12 * np.sin(2 * np.pi * (doy - 80) / 365)        # seasonal
        + 0.018 * (year - 1924)                              # warming trend
        + np.random.normal(0, 2.5, n_days)                   # daily noise
    )

    # Pressure (hPa)
    pressure = (
        1013
        - 5 * np.sin(2 * np.pi * (doy - 100) / 365)
        + np.random.normal(0, 4, n_days)
    )

    # Humidity (%)
    humidity = (
        60
        + 15 * np.sin(2 * np.pi * (doy - 120) / 365)
        + np.random.normal(0, 8, n_days)
    ).clip(20, 99)

    # Wind speed (km/h)
    wind_speed = np.abs(np.random.normal(15, 6, n_days))

    # Precipitation (mm) — skewed distribution
    precip = np.random.exponential(3, n_days) * (humidity / 100) ** 2

    # Cloud cover (0–1)
    cloud = (humidity / 100 * 0.7 + np.random.uniform(0, 0.3, n_days)).clip(0, 1)

    df = pd.DataFrame({
        "date":       dates,
        "doy":        doy,
        "month":      month,
        "year":       year,
        "temp":       base_temp.round(2),
        "pressure":   pressure.round(1),
        "humidity":   humidity.round(1),
        "wind_speed": wind_speed.round(1),
        "precip_mm":  precip.round(2),
        "cloud_cover":cloud.round(3),
    })

    # Lag features (past 1, 2, 3, 7 days)
    for lag in [1, 2, 3, 7]:
        df[f"temp_lag{lag}"]    = df["temp"].shift(lag)
        df[f"precip_lag{lag}"]  = df["precip_mm"].shift(lag)
        df[f"humid_lag{lag}"]   = df["humidity"].shift(lag)

    # Rolling statistics (7-day, 30-day)
    df["temp_roll7_mean"]   = df["temp"].rolling(7).mean()
    df["temp_roll7_std"]    = df["temp"].rolling(7).std()
    df["temp_roll30_mean"]  = df["temp"].rolling(30).mean()
    df["precip_roll7_sum"]  = df["precip_mm"].rolling(7).sum()

    # Targets
    df["next_temp"]     = df["temp"].shift(-1)          # regression target
    df["rain_tomorrow"] = (df["precip_mm"].shift(-1) > 1).astype(int)  # classification

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "doy", "month", "temp", "pressure", "humidity",
    "wind_speed", "precip_mm", "cloud_cover",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag7",
    "precip_lag1", "precip_lag2", "precip_lag3", "precip_lag7",
    "humid_lag1", "humid_lag2", "humid_lag3",
    "temp_roll7_mean", "temp_roll7_std",
    "temp_roll30_mean", "precip_roll7_sum",
]

# ─────────────────────────────────────────────
#  3. STACKING ENSEMBLE MODEL
# ─────────────────────────────────────────────
class WeatherStackingEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.base_learners = [
            ("random_forest",    RandomForestRegressor(n_estimators=120, max_depth=10,
                                                       min_samples_leaf=5, random_state=42, n_jobs=-1)),
            ("gradient_boost",   GradientBoostingRegressor(n_estimators=100, learning_rate=0.08,
                                                            max_depth=5, random_state=42)),
            ("xgboost",          xgb.XGBRegressor(n_estimators=150, learning_rate=0.07,
                                                   max_depth=6, subsample=0.8,
                                                   colsample_bytree=0.8, random_state=42,
                                                   verbosity=0)),
            ("svr",              SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
            ("ridge",            Ridge(alpha=1.0)),
        ]
        self.meta_learner = LinearRegression()
        self.stacking_model = StackingRegressor(
            estimators=self.base_learners,
            final_estimator=self.meta_learner,
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )
        self.metrics = {}
        self.feature_importance = {}
        self.is_trained = False

    def train(self, df):
        X = df[FEATURE_COLS].values
        y_temp = df["next_temp"].values

        X_scaled = self.scaler.fit_transform(X)

        # Train stacking model
        print("[AtmosAI] Training stacking ensemble on", len(X_scaled), "samples...")
        self.stacking_model.fit(X_scaled, y_temp)

        # Cross-validated metrics
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse = np.sqrt(-cross_val_score(
            self.stacking_model, X_scaled, y_temp,
            cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
        ))
        cv_r2 = cross_val_score(
            self.stacking_model, X_scaled, y_temp,
            cv=kf, scoring="r2", n_jobs=-1
        )

        y_pred = self.stacking_model.predict(X_scaled)
        self.metrics = {
            "rmse":        round(float(np.sqrt(mean_squared_error(y_temp, y_pred))), 3),
            "mae":         round(float(mean_absolute_error(y_temp, y_pred)), 3),
            "r2":          round(float(r2_score(y_temp, y_pred)), 4),
            "cv_rmse_mean":round(float(cv_rmse.mean()), 3),
            "cv_rmse_std": round(float(cv_rmse.std()), 3),
            "cv_r2_mean":  round(float(cv_r2.mean()), 4),
            "accuracy_pct":round(float(cv_r2.mean() * 100), 2),
        }

        # Per-model metrics (train each individually for comparison)
        individual = {}
        for name, estimator in self.base_learners:
            est = estimator.__class__(**estimator.get_params())
            scores = np.sqrt(-cross_val_score(
                est, X_scaled, y_temp,
                cv=3, scoring="neg_mean_squared_error", n_jobs=-1
            ))
            r2s = cross_val_score(est, X_scaled, y_temp, cv=3, scoring="r2", n_jobs=-1)
            individual[name] = {
                "rmse": round(float(scores.mean()), 3),
                "r2":   round(float(r2s.mean()), 4),
                "acc":  round(float(r2s.mean() * 100), 2),
            }
        individual["stacking_meta"] = {
            "rmse": self.metrics["cv_rmse_mean"],
            "r2":   self.metrics["cv_r2_mean"],
            "acc":  self.metrics["accuracy_pct"],
        }
        self.metrics["individual_models"] = individual

        # Feature importance (from Random Forest)
        rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y_temp)
        importances = rf.feature_importances_
        self.feature_importance = dict(
            sorted(zip(FEATURE_COLS, importances.tolist()),
                   key=lambda x: x[1], reverse=True)
        )

        self.is_trained = True
        print(f"[AtmosAI] Training complete. Stacking R²={self.metrics['r2']}, RMSE={self.metrics['rmse']}°C")
        return self.metrics

    def predict_next_days(self, latest_row_dict, n_days=7):
        """Predict next n_days temperature given latest conditions."""
        predictions = []
        current = {k: latest_row_dict.get(k, 0.0) for k in FEATURE_COLS}

        for day in range(n_days):
            x = np.array([[current[f] for f in FEATURE_COLS]])
            x_scaled = self.scaler.transform(x)
            pred_temp = float(self.stacking_model.predict(x_scaled)[0])
            
            # Simulate rain probability using physical heuristics
            rain_prob = self._rain_probability(current)
            rain_vol  = max(0, rain_prob * 40 * np.random.lognormal(0, 0.4))

            predictions.append({
                "day":        day + 1,
                "date":       (datetime.now() + timedelta(days=day+1)).strftime("%Y-%m-%d"),
                "weekday":    (datetime.now() + timedelta(days=day+1)).strftime("%A"),
                "temp_pred":  round(pred_temp, 1),
                "temp_high":  round(pred_temp + abs(np.random.normal(2.5, 0.8)), 1),
                "temp_low":   round(pred_temp - abs(np.random.normal(4, 1)), 1),
                "rain_prob":  round(rain_prob * 100, 1),
                "rain_vol_mm":round(rain_vol, 1),
                "condition":  self._condition_label(pred_temp, rain_prob, current["humidity"]),
                "icon":       self._condition_icon(pred_temp, rain_prob, current["humidity"]),
            })
            # Update rolling lags for next step
            current["temp_lag3"] = current["temp_lag2"]
            current["temp_lag2"] = current["temp_lag1"]
            current["temp_lag1"] = current["temp"]
            current["temp"]      = pred_temp
            current["precip_lag1"] = rain_vol
            doy_new = (current["doy"] % 365) + 1
            current["doy"] = doy_new
            current["month"] = max(1, min(12, int(doy_new / 30.4) + 1))

        return predictions

    def _rain_probability(self, ctx):
        h = ctx.get("humidity", 60) / 100
        p = ctx.get("precip_lag1", 0)
        c = ctx.get("cloud_cover", 0.5)
        score = 0.45 * h + 0.35 * min(p / 20, 1) + 0.20 * c
        return float(np.clip(score + np.random.normal(0, 0.05), 0, 1))

    def _condition_label(self, temp, rain_prob, humidity):
        if rain_prob > 0.8: return "Heavy Rain"
        if rain_prob > 0.55: return "Rainy"
        if rain_prob > 0.35: return "Showers"
        if humidity > 85: return "Foggy"
        if temp > 35: return "Scorching Hot"
        if temp > 28: return "Sunny & Hot"
        if temp > 18: return "Partly Cloudy"
        if temp > 8:  return "Cloudy"
        return "Cold & Overcast"

    def _condition_icon(self, temp, rain_prob, humidity):
        if rain_prob > 0.8:  return "⛈"
        if rain_prob > 0.55: return "🌧"
        if rain_prob > 0.35: return "🌦"
        if humidity > 85:    return "🌫"
        if temp > 35:        return "🔥"
        if temp > 28:        return "☀️"
        if temp > 18:        return "⛅"
        if temp > 8:         return "☁️"
        return "🥶"


# ─────────────────────────────────────────────
#  4. GLOBAL MODEL INIT
# ─────────────────────────────────────────────
print("[AtmosAI] Generating 100-year historical dataset...")
df_hist = generate_historical_dataset(n_years=100)

model = WeatherStackingEnsemble()
model_metrics = model.train(df_hist)


# ─────────────────────────────────────────────
#  5. FLASK ROUTES
# ─────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_trained": model.is_trained, "timestamp": datetime.now().isoformat()})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: { "city": "Mumbai", "lat": 19.07, "lon": 72.87,
            "temp": 31, "humidity": 88, "pressure": 1005,
            "wind_speed": 22, "precip_mm": 12, "cloud_cover": 0.9 }
    Returns: 7-day forecast + current conditions + risk analysis
    """
    data = request.get_json(force=True)

    # Build feature row from user input / defaults
    month = datetime.now().month
    doy   = datetime.now().timetuple().tm_yday
    current = {
        "doy":        doy,
        "month":      month,
        "temp":       float(data.get("temp", 25)),
        "pressure":   float(data.get("pressure", 1013)),
        "humidity":   float(data.get("humidity", 65)),
        "wind_speed": float(data.get("wind_speed", 12)),
        "precip_mm":  float(data.get("precip_mm", 2)),
        "cloud_cover":float(data.get("cloud_cover", 0.5)),
        # Lag features: use current as proxy (in real app, fetch from DB)
        "temp_lag1": float(data.get("temp", 25)) - 0.5,
        "temp_lag2": float(data.get("temp", 25)) - 1.0,
        "temp_lag3": float(data.get("temp", 25)) - 1.2,
        "temp_lag7": float(data.get("temp", 25)) - 2.0,
        "precip_lag1": float(data.get("precip_mm", 2)) * 0.9,
        "precip_lag2": float(data.get("precip_mm", 2)) * 0.7,
        "precip_lag3": float(data.get("precip_mm", 2)) * 0.5,
        "precip_lag7": float(data.get("precip_mm", 2)) * 0.3,
        "humid_lag1": float(data.get("humidity", 65)) - 1,
        "humid_lag2": float(data.get("humidity", 65)) - 2,
        "humid_lag3": float(data.get("humidity", 65)) - 2,
        "temp_roll7_mean":  float(data.get("temp", 25)) - 1,
        "temp_roll7_std":   2.1,
        "temp_roll30_mean": float(data.get("temp", 25)) - 0.5,
        "precip_roll7_sum": float(data.get("precip_mm", 2)) * 5,
    }

    forecast = model.predict_next_days(current, n_days=7)
    rain_p   = model._rain_probability(current)
    risk     = compute_disaster_risk(current, rain_p)
    hourly   = generate_hourly(current)

    return jsonify({
        "city":     data.get("city", "Unknown"),
        "current":  format_current(current),
        "forecast": forecast,
        "hourly":   hourly,
        "risk":     risk,
        "rain_probability_24h": round(rain_p * 100, 1),
        "rain_volume_24h_mm":   round(rain_p * 35, 1),
    })


@app.route("/api/metrics")
def metrics():
    """Returns model accuracy, RMSE, per-model scores, feature importance."""
    return jsonify({
        "overall": model_metrics,
        "feature_importance": dict(list(model.feature_importance.items())[:12]),
        "training_samples": len(df_hist),
        "data_years": 100,
        "features_used": len(FEATURE_COLS),
    })


@app.route("/api/historical")
def historical():
    """Returns aggregated historical stats for charts."""
    yearly = df_hist.groupby("year").agg(
        avg_temp=("temp", "mean"),
        max_temp=("temp", "max"),
        min_temp=("temp", "min"),
        total_precip=("precip_mm", "sum"),
    ).reset_index()

    monthly_avg = df_hist.groupby("month").agg(
        avg_temp=("temp", "mean"),
        avg_precip=("precip_mm", "mean"),
        avg_humidity=("humidity", "mean"),
    ).reset_index()

    records = {
        "hottest_day":  round(float(df_hist["temp"].max()), 1),
        "coldest_day":  round(float(df_hist["temp"].min()), 1),
        "max_rain_day": round(float(df_hist["precip_mm"].max()), 1),
        "max_wind":     round(float(df_hist["wind_speed"].max()), 1),
    }

    return jsonify({
        "yearly":     yearly.round(2).to_dict(orient="records"),
        "monthly":    monthly_avg.round(2).to_dict(orient="records"),
        "records":    records,
        "trend_slope_C_per_decade": round(
            float(np.polyfit(df_hist["year"].values, df_hist["temp"].values, 1)[0]) * 10, 3
        ),
    })


# ─────────────────────────────────────────────
#  6. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def format_current(ctx):
    return {
        "temp":       round(ctx["temp"], 1),
        "humidity":   round(ctx["humidity"], 1),
        "pressure":   round(ctx["pressure"], 1),
        "wind_speed": round(ctx["wind_speed"], 1),
        "precip_mm":  round(ctx["precip_mm"], 2),
        "cloud_cover":round(ctx["cloud_cover"], 2),
        "dew_point":  round(ctx["temp"] - ((100 - ctx["humidity"]) / 5), 1),
        "feels_like": round(ctx["temp"] - 0.4 * (1 - ctx["humidity"]/100) * (ctx["temp"] - 10), 1),
        "uv_index":   int(max(0, 8 * (1 - ctx["cloud_cover"]) * math.sin(math.pi * ctx["doy"] / 365))),
        "visibility_km": round(max(0.5, 15 * (1 - ctx["cloud_cover"] * 0.7) - ctx["precip_mm"] * 0.3), 1),
    }

def compute_disaster_risk(ctx, rain_p):
    h = ctx["humidity"] / 100
    w = ctx["wind_speed"] / 120
    p = ctx["precip_mm"] / 50
    pr7 = ctx["precip_roll7_sum"] / 200 if ctx.get("precip_roll7_sum") else p

    risks = {
        "flash_flood":  round(min(99, (rain_p * 0.5 + pr7 * 0.35 + h * 0.15) * 100), 1),
        "cyclone":      round(min(99, (w * 0.6 + rain_p * 0.3 + h * 0.1) * 100), 1),
        "lightning":    round(min(99, (rain_p * 0.55 + ctx["cloud_cover"] * 0.4 + h * 0.05) * 100), 1),
        "tornado":      round(min(99, (w * 0.7 + rain_p * 0.2 + abs(ctx["temp"] - ctx.get("temp_lag1", ctx["temp"])) * 0.02) * 100), 1),
        "drought":      round(min(99, max(0, (1 - rain_p) * 0.6 + (1 - h) * 0.4) * 100), 1),
        "heatwave":     round(min(99, max(0, (ctx["temp"] - 30) / 15 * 0.7 + (1 - h) * 0.3) * 100), 1),
    }
    def level(v):
        if v > 70: return "Extreme"
        if v > 50: return "High"
        if v > 25: return "Medium"
        return "Low"
    return {k: {"probability": v, "level": level(v)} for k, v in risks.items()}

def generate_hourly(ctx):
    base = ctx["temp"]
    hours = []
    for h in range(24):
        variation = 4 * math.sin(math.pi * (h - 6) / 12)
        t = base + variation + random.gauss(0, 0.5)
        rain_p = model._rain_probability({**ctx, "cloud_cover": ctx["cloud_cover"] + 0.1 * math.sin(h)})
        hours.append({
            "hour":     h,
            "time":     f"{h:02d}:00",
            "temp":     round(t, 1),
            "rain_prob":round(rain_p * 100, 1),
            "icon":     model._condition_icon(t, rain_p, ctx["humidity"]),
        })
    return hours


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AtmosAI Weather Prediction Engine — Running!")
    print(f"  Stacking R²  : {model_metrics['r2']}")
    print(f"  Stacking RMSE: {model_metrics['rmse']} °C")
    print(f"  Accuracy     : {model_metrics['accuracy_pct']}%")
    print("="*55)
    print("  API Endpoints:")
    print("  POST /api/predict   → 7-day forecast")
    print("  GET  /api/metrics   → model accuracy report")
    print("  GET  /api/historical → 100yr trend data")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
