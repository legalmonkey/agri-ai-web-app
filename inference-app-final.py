import os
import json
import json
print("json module imported successfully")

import math
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

PROJ_ROOT = r"C:\Users\Ritvik Bhat\agri ai web app"
MODEL_PATH = os.path.join(PROJ_ROOT, "artifacts_yield", "yield_pipeline_forward_chain_STACKED_PRODUCTION.pkl")
CROP_DURATIONS_PATH = os.path.join(PROJ_ROOT, "crop_durations.json")
ENRICHED_DATA_DIR = os.path.join(PROJ_ROOT, "processed_training_csvs")

templates = Jinja2Templates(directory="templates")

def _norm_text(s: str) -> str:
    return str(s).strip().lower().replace("&", "and").replace(".", "").replace("-", " ") if s else ""

def _load_model_and_resources():
    artifacts = joblib.load(MODEL_PATH)
    models = artifacts['models_forward_chain']['yieldcalc']
    meta_model = models['meta_model']
    fold_models = models['fold_models']
    crop_durations = json.load(open(CROP_DURATIONS_PATH))
    return meta_model, fold_models, crop_durations

_meta_model, _fold_models, _CROP_DURATIONS = _load_model_and_resources()

def _calculate_harvest_date_and_season(crop, sowing_date):
    dur_days = _CROP_DURATIONS.get(crop.strip(), {}).get("duration_days", 120)
    harvest_date = sowing_date + timedelta(days=dur_days)
    season = "Kharif" if 6 <= sowing_date.month <= 10 else "Rabi" if 11 <= sowing_date.month <= 2 else "Zaid"
    return harvest_date, dur_days, season

def _get_enriched_df():
    if not os.path.isdir(ENRICHED_DATA_DIR): return pd.DataFrame()
    run_dirs = [os.path.join(ENRICHED_DATA_DIR, d) for d in os.listdir(ENRICHED_DATA_DIR) if d.startswith("run_")]
    if not run_dirs: return pd.DataFrame()
    run_dirs.sort()
    enriched_csv = os.path.join(run_dirs[-1], "01_enriched_base.csv")
    df = pd.read_csv(enriched_csv)
    if "statenorm" not in df.columns:
        df["statenorm"] = df["statename"].map(_norm_text)
    if "districtnorm" not in df.columns:
        df["districtnorm"] = df["districtname"].map(_norm_text)
    return df

_enriched_df = _get_enriched_df()

def _get_lat_lon(state, district):
    df = _enriched_df
    if df.empty:
        return 22.9734, 78.6569  # generic center lat/lon
    st_nm, dt_nm = _norm_text(state), _norm_text(district)
    rec = df[(df["statenorm"]==st_nm) & (df["districtnorm"]==dt_nm)]
    if not rec.empty:
        lat, lon = rec.iloc[0]["lat"], rec.iloc[0]["lon"]
        if not pd.isnull(lat) and not pd.isnull(lon):
            return float(lat), float(lon)
    return 22.9734, 78.6569

def _fetch_weather_nasa(lat, lon, start_date, end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": "{:.4f}".format(lat),
        "longitude": "{:.4f}".format(lon),
        "start": start_str,
        "end": end_str,
        "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN",
        "community": "ag",
        "format": "JSON"
    }
    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        param = data.get("properties", {}).get("parameter", {})
        rain = param.get("PRECTOTCORR", {})
        tavg = param.get("T2M", {})
        tmax = param.get("T2M_MAX", {})
        tmin = param.get("T2M_MIN", {})
        df_weather = pd.DataFrame({"date": list(rain.keys())})
        df_weather["rain"] = df_weather["date"].map(rain)
        df_weather["tavg"] = df_weather["date"].map(tavg)
        df_weather["tmax"] = df_weather["date"].map(tmax)
        df_weather["tmin"] = df_weather["date"].map(tmin)
        df_weather["date"] = pd.to_datetime(df_weather["date"], format="%Y%m%d", errors="coerce")
        # Clean invalid data
        for col in ["rain","tavg","tmax","tmin"]:
            df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce")
            df_weather.loc[df_weather[col]<-900, col] = np.nan
        rainfall_sum = df_weather["rain"].sum()
        tavg_mean = df_weather["tavg"].mean()
        tmax_mean = df_weather["tmax"].mean()
        tmin_mean = df_weather["tmin"].mean()
        # Simplistic ET0 and GDD as placeholders
        et0_sum = 0.0
        gdd_sum = 0.0
        # Calculate GDD (simplified)
        df_weather["gdd"] = ((df_weather["tmax"] + df_weather["tmin"])/2 - 10).clip(lower=0)
        gdd_sum = df_weather["gdd"].sum()
        return {
            "Rainfall_sum": rainfall_sum,
            "Tavg_mean": tavg_mean,
            "Tmax_mean": tmax_mean,
            "Tmin_mean": tmin_mean,
            "ET0_sum": et0_sum,
            "GDD_sum": gdd_sum,
            "days_count": len(df_weather)
        }
    except:
        return {"Rainfall_sum":0,"Tavg_mean":0,"Tmax_mean":0,"Tmin_mean":0,"ET0_sum":0,"GDD_sum":0,"days_count":0}

def _get_lags(state, district, crop, year):
    df = _enriched_df
    if df.empty:
        return {"Rainfall_sum_lag1":600.0, "Tavg_mean_lag1":25.0, "Tmax_mean_lag1":30.0, "Tmin_mean_lag1":18.0, "ET0_sum_lag1":4.5, "GDD_sum_lag1":1400.0}
    sn, dn = _norm_text(state), _norm_text(district)
    crop_lower = crop.lower()
    hist = df[(df["statenorm"]==sn) & (df["districtnorm"]==dn) & (df["crop"].str.lower()==crop_lower) & (df["cropyear"]==year-1)]
    lags = {}
    for col in ["Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
        lag_col = f"{col}_lag1"
        if not hist.empty and col in hist.columns:
            lags[lag_col] = float(hist.iloc[0][col])
        else:
            defaults = {"Rainfall_sum":600,"Tavg_mean":25,"Tmax_mean":30,"Tmin_mean":18,"ET0_sum":4.5,"GDD_sum":1400}
            lags[lag_col] = defaults.get(col, 0)
    return lags

def _analyze_confidence(stats, season_progress, lags, std_dev):
    confs = {
        "base_confidence":0.50,
        "season_progress_bonus": season_progress*0.30,
        "model_agreement_factor": 0.20 if std_dev<0.3 else (0.10 if std_dev<0.6 else (-0.15 if std_dev>=1.0 else 0.0)),
        "weather_adjustment":0.0,
        "historical_data_bonus":0.10 if lags.get('Rainfall_sum_lag1',0)>0 else 0.0
    }
    rain = stats["Rainfall_sum"]
    days_count = stats.get("days_count", 90)
    risks = []
    recs = []
    exp_rain = days_count*5
    if rain < exp_rain*0.6:
        confs["weather_adjustment"] -= 0.10
        risks.append(f"Low rainfall: {rain:.1f}mm")
        recs.append("Supplemental irrigation recommended")
    confidence_score = max(0.20, min(0.95, sum(confs.values())))
    confidence_level = "High" if confidence_score>=0.75 else "Medium" if confidence_score>=0.55 else "Low"
    confidence_explanation = {
        "High":"Strong model agreement and favorable weather",
        "Medium":"Moderate confidence due to weather or model variation",
        "Low":"Low confidence due to significant concerns"
    }[confidence_level]
    return {
        "confidence_score":confidence_score,
        "confidence_level":confidence_level,
        "confidence_explanation":confidence_explanation,
        "confidence_breakdown":confs,
        "risks":risks,
        "recommendations":recs
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    html = '''
    <html><head><title>AgriAI Yield Predictor</title></head><body>
    <h1>AgriAI Yield Predictor</h1>
    <form action="/predict" method="post">
    State: <input name="state" value="Punjab"><br>
    District: <input name="district" value="Ludhiana"><br>
    Crop: <input name="crop" value="Wheat"><br>
    Area (ha): <input name="area" value="2.5" type="number" step="0.01"><br>
    Sowing date (YYYY-MM-DD): <input name="sowing_date" value="2024-09-15"><br>
    Forecast date (YYYY-MM-DD, optional): <input name="forecast_date"><br>
    <input type="submit" value="Predict Yield">
    </form></body></html>
    '''
    return HTMLResponse(content=html)

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    area: float = Form(...),
    sowing_date: str = Form(...),
    forecast_date: str = Form(None)
):
    sow_dt = datetime.strptime(sowing_date, "%Y-%m-%d")
    fore_dt = datetime.strptime(forecast_date, "%Y-%m-%d") if forecast_date else datetime.now()

    lat, lon = _get_lat_lon(state, district)

    harvest_date, dur_days, season = _calculate_harvest_date_and_season(crop, sow_dt)

    weather_stats = _fetch_weather_nasa(lat, lon, sow_dt, fore_dt)
    lags = _get_lags(state, district, crop, sow_dt.year)

    # Compose feature dict for inference
    feature_dict = {
      "statename": state,
      "districtname": district,
      "statenorm": _norm_text(state),
      "districtnorm": _norm_text(district),
      "crop": crop,
      "season": season,
      "cropyear": sow_dt.year,
      "area": area,
      "lat": lat,
      "lon": lon,
      "Rainfall_sum": weather_stats["Rainfall_sum"],
      "Tavg_mean": weather_stats["Tavg_mean"],
      "Tmax_mean": weather_stats["Tmax_mean"],
      "Tmin_mean": weather_stats["Tmin_mean"],
      "ET0_sum": weather_stats["ET0_sum"],
      "GDD_sum": weather_stats["GDD_sum"],
      **lags,
      **{ f"{k}_delta1": weather_stats.get(k.replace("_lag1", ""), 0) - v for k,v in lags.items() if k.endswith("_lag1")}
    }

    df_input = pd.DataFrame([feature_dict])

    fold_preds = []
    errors = []
    for i, fold in enumerate(_fold_models):
        try:
            pred = fold.predict(df_input)[0]
            fold_preds.append(pred)
        except Exception as e:
            errors.append(f"Fold {i} prediction error: {e}")

    if not fold_preds:
        pred_yield = 1.0
    else:
        preds_arr = np.array(fold_preds)
        mean_pred = preds_arr.mean()
        meta_feat = np.hstack([mean_pred.reshape(1,1), preds_arr.reshape(1, -1)])
        raw_pred = _meta_model.predict(meta_feat)[0]
        pred_yield = max(0.5, min(raw_pred, 15.0))

    season_progress, days_total, days_elapsed, days_remaining, _ = \
        _season_progress_with_sowing(crop, sow_dt, fore_dt)

    conf_info = _analyze_confidence(weather_stats, season_progress, lags, np.std(fold_preds) if fold_preds else 0)

    # Display minimal styled output
    html = f"""
    <html><body>
    <h1>Yield Prediction Result</h1>
    <p><b>State:</b> {state}</p>
    <p><b>District:</b> {district}</p>
    <p><b>Crop:</b> {crop}</p>
    <p><b>Area (ha):</b> {area}</p>
    <p><b>Sowing Date:</b> {sowing_date}</p>
    <p><b>Forecast Date:</b> {forecast_date or fore_dt.strftime('%Y-%m-%d')}</p>
    <p><b>Predicted Yield (t/ha):</b> {pred_yield:.3f}</p>
    <p><b>Season Progress:</b> {season_progress*100:.1f}%</p>
    <p><b>Days Elapsed:</b> {days_elapsed}</p>
    <p><b>Days Remaining:</b> {days_remaining}</p>
    <p><b>Confidence Level:</b> {conf_info['confidence_level']}</p>
    <p><b>Confidence Explanation:</b> {conf_info['confidence_explanation']}</p>
    <p><b>Model Fold Predictions:</b> {fold_preds}</p>
    <p><b>Errors (if any):</b> {errors}</p>
    <a href="/">Back</a>
    </body></html>"""
    return HTMLResponse(content=html)
