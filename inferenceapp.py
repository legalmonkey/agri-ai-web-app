from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

import os
import json
import math
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd
import joblib
import requests

warnings.filterwarnings("ignore")

app = FastAPI()

# ------------------------------ Project paths ------------------------------
PROJ_ROOT = os.getcwd()
MODEL_PATH = os.path.join(PROJ_ROOT, "artifacts_yield", "yield_pipeline.pkl")
CROP_DURATIONS_PATH = os.path.join(PROJ_ROOT, "crop_durations.json")
RULES_PATH = os.path.join(PROJ_ROOT, "rules", "crop_reco.json")
PROC_DIR = os.path.join(PROJ_ROOT, "processed_training_csvs")

# ------------------------------ Constants ------------------------------
DROUGHT_THRESHOLD = 50.0
HEAT_STRESS_THRESHOLD = 35.0
OPTIMAL_TEMP_RANGE = (20.0, 30.0)

# ------------------------------ Globals ------------------------------
_ready = {"ok": False, "reason": "initializing"}
_preprocessor = None
_models_xgb = None
_meta_models = None
_targets = []
_CROP_DURATIONS: Dict = {}
_RULES_CACHE: Dict = {}
_ENRICHED_DF: Optional[pd.DataFrame] = None

# ------------------------------ Utilities ------------------------------
def _norm_text(s: str) -> str:
    return str(s).strip().lower().replace("&", "and").replace(".", "").replace("-", " ") if s is not None else ""

def _latest_enriched_path() -> Optional[str]:
    if not os.path.isdir(PROC_DIR):
        return None
    runs = [os.path.join(PROC_DIR, d) for d in os.listdir(PROC_DIR) if d.startswith("run_")]
    if not runs:
        return None
    runs.sort()
    cand = os.path.join(runs[-1], "01_enriched_base.csv")
    return cand if os.path.isfile(cand) else None

def _load_enriched_base() -> pd.DataFrame:
    global _ENRICHED_DF
    if _ENRICHED_DF is not None:
        return _ENRICHED_DF
    path = _latest_enriched_path()
    if path and os.path.isfile(path):
        df = pd.read_csv(path)
        if "statenorm" not in df.columns and "statename" in df.columns:
            df["statenorm"] = df["statename"].astype(str).map(_norm_text)
        if "districtnorm" not in df.columns and "districtname" in df.columns:
            df["districtnorm"] = df["districtname"].astype(str).map(_norm_text)
        _ENRICHED_DF = df
        return df
    _ENRICHED_DF = pd.DataFrame()
    return _ENRICHED_DF

def _infer_season_from_sowing_date(sowing_date):
    """Infer Kharif/Rabi/Zaid based on sowing month"""
    month = sowing_date.month
    if 5 <= month <= 7:  # May-July
        return "Kharif"
    elif 10 <= month <= 12 or month <= 2:  # Oct-Feb
        return "Rabi"
    else:  # Mar-Apr
        return "Zaid"

def _calculate_harvest_date(crop, sowing_date):
    """Calculate harvest date based on crop-specific duration"""
    crop_key = crop.strip()
    
    if crop_key in _CROP_DURATIONS:
        duration_days = _CROP_DURATIONS[crop_key]["duration_days"]
    else:
        duration_days = 120
    
    harvest_date = sowing_date + timedelta(days=duration_days)
    season_inferred = _infer_season_from_sowing_date(sowing_date)
    
    return harvest_date, duration_days, season_inferred

def _season_progress_with_sowing(crop, sowing_date, current_date=None):
    """Calculate season progress based on sowing date and crop-specific duration"""
    if current_date is None:
        current_date = datetime.now()
    
    harvest_date, duration_days, season_inferred = _calculate_harvest_date(crop, sowing_date)
    
    days_elapsed = (current_date - sowing_date).days
    days_remaining = (harvest_date - current_date).days
    
    past_harvest = current_date > harvest_date
    
    if past_harvest:
        progress = 1.0
        days_remaining = 0
    else:
        progress = max(0.0, min(1.0, days_elapsed / duration_days))
    
    return {
        'progress': progress,
        'harvest_date': harvest_date,
        'days_total': duration_days,
        'days_elapsed': days_elapsed,
        'days_remaining': max(0, days_remaining),
        'season_inferred': season_inferred,
        'past_harvest': past_harvest
    }

# ------------------------------ Rules loader ------------------------------
def load_crop_rules() -> dict:
    global _RULES_CACHE
    if _RULES_CACHE:
        return _RULES_CACHE
    if not os.path.isfile(RULES_PATH):
        _RULES_CACHE = {
            "rice": {
                "fertilizer_blend": {"npk": "NPK 18-46-0 + 0-0-60 (split)", "note": "Adjust by soil test; split N and K."},
                "irrigation": {"title": "Sprinkler System", "subtitle": "Optimized for cereals/loams"},
                "pesticides": ["Imidacloprid", "Fipronil", "Chlorantraniliprole"]
            }
        }
        return _RULES_CACHE
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        _RULES_CACHE = json.load(f)
    return _RULES_CACHE

def recommend_for_crop(crop: str) -> dict:
    rules = load_crop_rules()
    rule = rules.get(_norm_text(crop))
    if not rule:
        rule = {
            "fertilizer_blend": {"npk": "NPK 18-46-0 (DAP) + Urea split", "note": "Refine with soil test maps"},
            "irrigation": {"title": "Sprinkler System", "subtitle": "Uniform coverage"},
            "pesticides": ["Imidacloprid", "Mancozeb", "Chlorantraniliprole"]
        }
    return rule

# ------------------------------ Georesolver ------------------------------
def resolve_lat_lon(state: str, district: str):
    df = _load_enriched_base()
    if df.empty:
        return (22.9734, 78.6569)
    st, dt = _norm_text(state), _norm_text(district)
    cand = df[(df.get("statenorm","")==st) & (df.get("districtnorm","")==dt)]
    if "lat" in df.columns and "lon" in df.columns and not cand.empty:
        lat = pd.to_numeric(cand["lat"], errors="coerce").dropna()
        lon = pd.to_numeric(cand["lon"], errors="coerce").dropna()
        if not lat.empty and not lon.empty:
            return float(lat.iloc[0]), float(lon.iloc[0])
    st_rows = df[df.get("statenorm","")==st]
    if not st_rows.empty and {"lat","lon"}.issubset(st_rows.columns):
        lat = pd.to_numeric(st_rows["lat"], errors="coerce").dropna()
        lon = pd.to_numeric(st_rows["lon"], errors="coerce").dropna()
        if not lat.empty and not lon.empty:
            return float(lat.iloc[0]), float(lon.iloc[0])
    return (22.9734, 78.6569)

# ------------------------------ NASA POWER ------------------------------
def fetch_past_week_weather(lat: float, lon: float, end_date: Optional[str]) -> Dict[str, float]:
    if end_date:
        try:
            ref = datetime.strptime(end_date, "%Y-%m-%d")
        except Exception:
            ref = datetime.utcnow()
    else:
        ref = datetime.utcnow()
    
    end_str = ref.strftime("%Y%m%d")
    start_str = (ref - timedelta(days=8)).strftime("%Y%m%d")

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start_str,
        "end": end_str,
        "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN",
        "community": "ag",
        "temporal": "daily",
        "format": "JSON"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        param = data.get("properties", {}).get("parameter", {})
        
        pre = param.get("PRECTOTCORR", {})
        tavg = param.get("T2M", {})
        tmax = param.get("T2M_MAX", {})
        tmin = param.get("T2M_MIN", {})

        keys = list((pre or tavg or tmax or tmin).keys())
        if not keys:
            return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0}

        df = pd.DataFrame({"date": keys})
        for k, series in [("rain", pre), ("tavg", tavg), ("tmax", tmax), ("tmin", tmin)]:
            df[k] = df["date"].map(series) if isinstance(series, dict) else np.nan

        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        for c in ["rain","tavg","tmax","tmin"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < -900, c] = np.nan

        exact_end = pd.to_datetime(end_str, format="%Y%m%d")
        exact_start = exact_end - pd.Timedelta(days=7)
        df = df[(df["date"] > exact_start) & (df["date"] <= exact_end)]

        if df.empty:
            return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0}

        tmax_vals = df["tmax"].dropna()
        tmin_vals = df["tmin"].dropna()
        tavg_vals = df["tavg"].dropna()
        
        et0_sum = 0.0
        gdd_sum = 0.0
        if not tmax_vals.empty and not tmin_vals.empty:
            for i in range(len(df)):
                if pd.notna(df.iloc[i]["tmax"]) and pd.notna(df.iloc[i]["tmin"]) and pd.notna(df.iloc[i]["tavg"]):
                    tmx = df.iloc[i]["tmax"]
                    tmn = df.iloc[i]["tmin"]
                    tav = df.iloc[i]["tavg"]
                    et0_sum += 0.0023 * max(0, tav - 17.8) * math.sqrt(max(0, tmx - tmn))
            gdd_daily = ((tmax_vals + tmin_vals) / 2 - 10.0).clip(lower=0)
            gdd_sum = float(gdd_daily.sum())

        return {
            "Rainfall_sum": float(df["rain"].sum(skipna=True)),
            "Tavg_mean": float(tavg_vals.mean()) if not tavg_vals.empty else 0.0,
            "Tmax_mean": float(tmax_vals.mean()) if not tmax_vals.empty else 0.0,
            "Tmin_mean": float(tmin_vals.mean()) if not tmin_vals.empty else 0.0,
            "ET0_sum": et0_sum,
            "GDD_sum": gdd_sum,
        }
    except Exception:
        return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0}

# ------------------------------ Historical lags ------------------------------
def _get_historical_lags(state, district, crop, cropyear):
    """Fetch 1-year lag values from enriched base"""
    df = _load_enriched_base()
    if df.empty:
        lags = {}
        for col in ["yieldcalc", "production", "area", "Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
            lags[f"{col}_lag1"] = 0.0
        return lags
    
    sn, dn = _norm_text(state), _norm_text(district)
    crop_norm = crop.lower().strip()
    
    lag_year = cropyear - 1
    hist = df[(df["statenorm"]==sn) & (df["districtnorm"]==dn) & 
              (df["crop"].str.lower().str.strip() == crop_norm) & 
              (df["cropyear"] == lag_year)]
    
    lags = {}
    if not hist.empty:
        for col in ["yieldcalc", "production", "area", "Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
            if col in hist.columns:
                lags[f"{col}_lag1"] = float(hist[col].iloc[0])
            else:
                lags[f"{col}_lag1"] = 0.0
    else:
        for col in ["yieldcalc", "production", "area", "Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
            lags[f"{col}_lag1"] = 0.0
    
    return lags

# ------------------------------ Risk assessment ------------------------------
def assess_weather_risks(week, progress):
    risks = {
        'drought_risk': False, 
        'heat_stress_risk': False, 
        'optimal_conditions': True, 
        'risk_level': 'LOW', 
        'advisory': ''
    }
    advisories = []
    
    if week['Rainfall_sum'] < DROUGHT_THRESHOLD:
        risks['drought_risk'] = True
        risks['optimal_conditions'] = False
        advisories.append(f"⚠ DROUGHT: {week['Rainfall_sum']:.1f}mm in past week")
    
    if week['Tmax_mean'] > HEAT_STRESS_THRESHOLD:
        risks['heat_stress_risk'] = True
        risks['optimal_conditions'] = False
        advisories.append(f"⚠ HEAT STRESS: Max temp {week['Tmax_mean']:.1f}°C")
    
    if risks['drought_risk'] and risks['heat_stress_risk']:
        risks['risk_level'] = 'HIGH'
    elif risks['drought_risk'] or risks['heat_stress_risk']:
        risks['risk_level'] = 'MODERATE'
    else:
        if risks['optimal_conditions']:
            advisories.append("✓ OPTIMAL: Conditions favorable")
    
    risks['advisory'] = "\n".join(advisories) if advisories else "No risks"
    return risks

# ------------------------------ Model load ------------------------------
def _safe_load_model():
    global _preprocessor, _models_xgb, _meta_models, _targets, _ready, _CROP_DURATIONS
    try:
        # Load ML pipeline
        artifacts = joblib.load(MODEL_PATH)
        _preprocessor = artifacts['preprocessor']
        _models_xgb = artifacts['models_xgb']
        _meta_models = artifacts['meta_models']
        _targets = artifacts['targets']
        
        # Load crop durations
        if os.path.isfile(CROP_DURATIONS_PATH):
            with open(CROP_DURATIONS_PATH, 'r') as f:
                _CROP_DURATIONS = json.load(f)
        else:
            _CROP_DURATIONS = {}
        
        _ready["ok"] = True
        _ready["reason"] = "ready"
    except Exception as e:
        _ready["ok"] = False
        _ready["reason"] = f"startup failed: {e}"

_safe_load_model()

# ------------------------------ Core prediction ------------------------------
def _predict_core(state: str, district: str, crop: str, land_area: float, sowing_date: str, end_date: Optional[str]):
    if not _ready.get("ok", False):
        raise RuntimeError(f"Model not ready: {_ready.get('reason')}")

    # Parse dates
    sowing_dt = datetime.strptime(sowing_date, "%Y-%m-%d")
    ref_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    cropyear = sowing_dt.year

    # Calculate season progress
    season_info = _season_progress_with_sowing(crop, sowing_dt, ref_date)
    season = season_info['season_inferred']

    # Get coordinates
    lat, lon = resolve_lat_lon(state, district)
    
    # Fetch weather
    week = fetch_past_week_weather(lat, lon, end_date=end_date)
    
    # Get historical lags
    lags = _get_historical_lags(state, district, crop, cropyear)
    
    # Compute deltas
    deltas = {}
    for base_col in ["Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
        deltas[f"{base_col}_delta1"] = week[base_col] - lags[f"{base_col}_lag1"]
    for base_col in ["yieldcalc", "production", "area"]:
        deltas[f"{base_col}_delta1"] = 0.0
    
    # Build input
    input_data = pd.DataFrame([{
        'statename': state,
        'districtname': district,
        'statenorm': _norm_text(state),
        'districtnorm': _norm_text(district),
        'crop': crop,
        'season': season,
        'cropyear': cropyear,
        'area': float(land_area),
        'lat': lat,
        'lon': lon,
        'Rainfall_sum': week['Rainfall_sum'],
        'Tavg_mean': week['Tavg_mean'],
        'Tmax_mean': week['Tmax_mean'],
        'Tmin_mean': week['Tmin_mean'],
        'ET0_sum': week['ET0_sum'],
        'GDD_sum': week['GDD_sum'],
        'Rainfall_sum_x': week['Rainfall_sum'],
        'Tavg_mean_x': week['Tavg_mean'],
        'Tmax_mean_x': week['Tmax_mean'],
        'Tmin_mean_x': week['Tmin_mean'],
        'ET0_sum_x': week['ET0_sum'],
        'GDD_sum_x': week['GDD_sum'],
        'Rainfall_sum_y': week['Rainfall_sum'],
        'Tavg_mean_y': week['Tavg_mean'],
        'Tmax_mean_y': week['Tmax_mean'],
        'Tmin_mean_y': week['Tmin_mean'],
        'ET0_sum_y': week['ET0_sum'],
        'GDD_sum_y': week['GDD_sum'],
        **lags,
        **deltas,
    }])
    
    # Preprocess
    X_proc = _preprocessor.transform(input_data)
    
    # Get predictions
    yield_idx = _targets.index('yieldcalc')
    
    all_target_preds = []
    for t_idx in range(len(_targets)):
        fold_preds = np.array([m.predict(X_proc) for m in _models_xgb[t_idx]])
        target_avg = np.mean(fold_preds, axis=0)
        all_target_preds.append(target_avg[0])
    
    stack_input = np.array(all_target_preds).reshape(1, -1)
    yield_pred = _meta_models[yield_idx].predict(stack_input)[0]
    
    # Clip to realistic bounds
    yield_pred = max(0.1, min(yield_pred, 20.0))
    prod_pred = yield_pred * land_area
    
    # Risk assessment
    risk_assessment = assess_weather_risks(week, season_info['progress'])
    
    confidence_score = 0.5 + (season_info['progress'] * 0.5)

    return {
        "state": state,
        "district": district,
        "crop": crop,
        "season": season,
        "lat": lat,
        "lon": lon,
        "sowing_date": sowing_dt.strftime("%Y-%m-%d"),
        "harvest_date": season_info['harvest_date'].strftime("%Y-%m-%d"),
        "crop_duration_days": season_info['days_total'],
        "yield_per_hectare": yield_pred,
        "area_input": land_area,
        "production_pred": prod_pred,
        "forecast_date": ref_date.strftime("%Y-%m-%d"),
        "season_progress_pct": round(season_info['progress'] * 100, 1),
        "days_elapsed": season_info['days_elapsed'],
        "days_remaining": season_info['days_remaining'],
        "past_harvest": season_info['past_harvest'],
        "confidence_score": int(confidence_score * 100),
        "weather": week,
        "risk": risk_assessment,
    }

# ------------------------------ UI Routes ------------------------------
@app.get("/", response_class=HTMLResponse)
def form():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agricultural Analysis</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg:#070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
      --accent:#30d158; --text:#e6f5ea; --muted:#a9b8ae;
      --input-bg: rgba(255,255,255,0.06); --input-border: rgba(255,255,255,0.12);
      --input-focus: rgba(48, 209, 88, 0.55);
    }
    *{ box-sizing:border-box; }
    html,body{
      height:100%; margin:0;
      font-family:"Inter",system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg);
      color:var(--text);
    }
    .container{ min-height: 100%; width: 100%; display: flex; align-items: flex-start; justify-content: center; padding: 48px 24px; }
    .panel{
      width: 100%; max-width: 1200px; padding: 46px 36px 40px; border-radius: 18px; background: var(--card);
      border: 1px solid var(--card-border); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
      box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06); position: relative;
    }
    .home-btn{
      position: absolute; top: 18px; right: 18px; display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:12px; color: var(--text);
      text-decoration:none; background: rgba(48, 209, 88, 0.12); border:1px solid rgba(48, 209, 88, 0.22); transition: all .2s ease;
    }
    h1{ margin:6px 0 8px; font-size: clamp(32px, 3.6vw, 56px); text-align: center; font-weight: 800; }
    .title-plain { color: #ecfff3; } .title-accent { color: var(--accent); }
    .subtitle{ margin: 0 0 30px; text-align:center; color: var(--muted); font-size: 16px; }

    form{ display:grid; gap: 18px; margin-top: 8px; }
    .field{ display:flex; flex-direction:column; gap:10px; }
    .label{ display:flex; align-items:center; gap:10px; font-weight:600; color:#d6edde; letter-spacing:.2px; }
    .label small{ color:#9eb1a6; font-weight:500; }

    .input{
      width:100%; padding:16px 16px; border-radius:12px; border:1px solid var(--input-border); background: var(--input-bg);
      color: var(--text); outline:none; transition: border-color .2s, box-shadow .2s, background .2s; font-size:16px;
    }
    .input::placeholder{ color:#94a89c; }
    .input:focus{ border-color:var(--input-focus); box-shadow:0 0 0 4px rgba(48,209,88,0.15); background:rgba(255,255,255,0.09); }

    .row{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }

    .btn{
      margin-top: 8px; padding: 18px 20px; width:100%; border:none; border-radius:14px; font-size:18px; font-weight:700; color:#052d14;
      background: linear-gradient(90deg, #28d17a 0%, #30d158 45%, #28d17a 100%); cursor:pointer; transition: transform .08s, filter .2s, box-shadow .2s;
      box-shadow: 0 10px 24px rgba(48,209,88,0.25), inset 0 1px 0 rgba(255,255,255,0.35);
    }
    .btn:hover{ filter: brightness(1.03); }
    .btn:active{ transform: translateY(1px); }

    select.input{
      color:#e6f5ea; background-color: rgba(255,255,255,0.06);
      max-width: 100%;
      border-color: var(--input-border);
    }
    select.input option{
      color:#0b130f; background:#ffffff;
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="panel" aria-label="Agricultural Analysis Form">
      <a class="home-btn" href="#"><span aria-hidden="true">🏠</span><span>Home</span></a>

      <h1><span class="title-plain">Agricultural</span><span class="title-accent"> Analysis</span></h1>
      <p class="subtitle">Enter farm details to receive AI‑powered insights</p>

      <form id="farmForm" method="post" action="/predict">
        <div class="row">
          <div class="field">
            <label class="label" for="state"><span class="dot"></span> State</label>
            <input class="input" id="state" name="state" type="text" placeholder="e.g., Punjab" required />
          </div>
          <div class="field">
            <label class="label" for="district"><span class="dot"></span> District</label>
            <input class="input" id="district" name="district" type="text" placeholder="e.g., Ludhiana" required />
          </div>
        </div>

        <div class="field">
          <label class="label" for="crop"><span class="dot"></span> Crop Type</label>
          <select class="input" id="crop" name="crop" required>
            <option value="" disabled selected>Select a crop</option>
            <option>Arecanut</option>
            <option>Arhar/Tur</option>
            <option>Bajra</option>
            <option>Banana</option>
            <option>Barley</option>
            <option>Black pepper</option>
            <option>Cardamom</option>
            <option>Cashewnut</option>
            <option>Castor seed</option>
            <option>Coconut</option>
            <option>Coriander</option>
            <option>Cotton(lint)</option>
            <option>Cowpea(Lobia)</option>
            <option>Dry chillies</option>
            <option>Dry ginger</option>
            <option>Garlic</option>
            <option>Ginger</option>
            <option>Gram</option>
            <option>Groundnut</option>
            <option>Guar seed</option>
            <option>Horse-gram</option>
            <option>Jowar</option>
            <option>Jute</option>
            <option>Khesari</option>
            <option>Linseed</option>
            <option>Maize</option>
            <option>Mango</option>
            <option>Masoor</option>
            <option>Mesta</option>
            <option>Moong(Green Gram)</option>
            <option>Moth</option>
            <option>Niger seed</option>
            <option>Oilseeds total</option>
            <option>Onion</option>
            <option>Other  Rabi pulses</option>
            <option>Other Cereals & Millets</option>
            <option>Other Kharif pulses</option>
            <option>Peas & beans (Pulses)</option>
            <option>Potato</option>
            <option>Ragi</option>
            <option>Rapeseed &Mustard</option>
            <option>Rice</option>
            <option>Safflower</option>
            <option>Sannhamp</option>
            <option>Sesamum</option>
            <option>Small millets</option>
            <option>Soyabean</option>
            <option>Sugarcane</option>
            <option>Sunflower</option>
            <option>Sweet potato</option>
            <option>Tapioca</option>
            <option>Tobacco</option>
            <option>Turmeric</option>
            <option>Urad</option>
            <option>Wheat</option>
            <option>other oilseeds</option>
          </select>
        </div>

        <div class="row">
          <div class="field">
            <label class="label" for="land"><span class="dot"></span> Land Size (Hectares)</label>
            <input class="input" id="land" name="land_area" type="number" inputmode="decimal" step="0.01" min="0" placeholder="Enter land size in hectares" required />
          </div>
          <div class="field">
            <label class="label" for="sowingDate"><span class="dot"></span> Sowing Date</label>
            <input class="input" id="sowingDate" name="sowing_date" type="text" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="\\d{4}-\\d{2}-\\d{2}" title="Enter date as YYYY-MM-DD (e.g., 2025-08-10)" required />
          </div>
        </div>

        <div class="field">
          <label class="label" for="endDate"><span class="dot"></span> Today's Date <small>(for weather data)</small></label>
          <input class="input" id="endDate" name="end_date" type="text" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="\\d{4}-\\d{2}-\\d{2}" title="Enter date as YYYY-MM-DD (e.g., 2025-10-21)" />
        </div>

        <button class="btn" type="submit">Run AI Analysis</button>
      </form>
    </section>
  </div>
</body>
</html>
    """

@app.get("/health", response_class=HTMLResponse)
def health():
    return "ok"

# ------------------------------ Results page ------------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    land_area: float = Form(...),
    sowing_date: str = Form(...),
    end_date: Optional[str] = Form(None)
):
    try:
        out = _predict_core(state, district, crop, land_area, sowing_date, end_date)

        def _f2(x):
            try: return f"{float(x):,.2f}"
            except: return str(x)
        def _f3(x):
            try: return f"{float(x):,.3f}"
            except: return str(x)

        # Recommendations
        reco = recommend_for_crop(out["crop"])
        irr_title = reco["irrigation"]["title"]
        irr_sub   = reco["irrigation"]["subtitle"]
        fert_npk  = reco["fertilizer_blend"]["npk"]
        fert_note = reco["fertilizer_blend"]["note"]
        pest_active_list = reco.get("pesticides", [])
        pest_grid_items = "".join([f'<div class="pill">{a}</div>' for a in (pest_active_list if pest_active_list else ["No actives"])])

        # Risk level color
        risk_level = out['risk']['risk_level']
        if risk_level == 'LOW':
            risk_color = '#30d158'
            risk_bg = 'rgba(48, 209, 88, 0.12)'
            risk_border = 'rgba(48, 209, 88, 0.25)'
        elif risk_level == 'MODERATE':
            risk_color = '#ff9f0a'
            risk_bg = 'rgba(255, 159, 10, 0.12)'
            risk_border = 'rgba(255, 159, 10, 0.25)'
        else:  # HIGH
            risk_color = '#ff453a'
            risk_bg = 'rgba(255, 69, 58, 0.12)'
            risk_border = 'rgba(255, 69, 58, 0.25)'

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Analysis Complete</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{{ --bg:#070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
       --accent:#30d158; --text:#e6f5ea; --muted:#a9b8ae; }}
*{{ box-sizing:border-box; }}
html,body{{ height:100%; margin:0; font-family:"Inter",system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
           background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg); color:var(--text); }}
.container{{ min-height:100%; width:100%; padding:40px 24px 60px; }}
.panel{{ width:100%; max-width:1200px; margin: 0 auto; padding: 0 6px; }}

.header{{ text-align:center; margin-bottom: 26px; position: relative; }}
.top-home{{ position:absolute; top:-8px; right:0; }}
.home-btn-small{{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:12px; color:#e6f5ea;
                 text-decoration:none; background: rgba(48, 209, 88, 0.12); border:1px solid rgba(48, 209, 88, 0.22); }}

.h-title{{ font-size: clamp(32px, 4.5vw, 58px); font-weight: 800; letter-spacing:.3px; }}
.h-title .accent{{ color: var(--accent); }}
.h-sub{{ margin-top:8px; color:#bcd4c5; }}

.grid-2{{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
.card{{ background: var(--card); border:1px solid var(--card-border); border-radius:18px; padding:22px;
        box-shadow:0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06); }}
.card h3{{ margin:4px 0 12px; font-size:18px; color:#dff2e6; }}
.kv{{ display:flex; justify-content:space-between; gap:12px; padding:10px 0; border-top:1px solid rgba(255,255,255,0.06); }}
.kv:first-of-type{{ border-top:none; }}
.k{{ color:#d0e5da; }} .v{{ color:#bfe9cc; font-weight:600; }}

.yield-big{{ font-size: clamp(28px, 4vw, 46px); font-weight:800; color:#c4ffd2; }}
.yield-sub{{ margin-top:6px; color:#9cc0b1; font-weight:600; }}

.pest-grid{{ margin-top:8px; display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }}
.pill{{ text-align:center; padding:10px 12px; border-radius:12px; background:rgba(155,125,200,.12);
       border:1px solid rgba(155,125,200,.3); color:#e7d9ff; font-weight:700; }}

.progress-card{{ margin-top:18px; }}
.progress-bar{{ width:100%; height:12px; background:rgba(255,255,255,0.08); border-radius:12px; overflow:hidden; margin-top:12px; }}
.progress-fill{{ height:100%; background: linear-gradient(90deg, #28d17a, #30d158); border-radius:12px; transition:width 0.6s ease; }}

.risk-badge{{ display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:12px; font-weight:700; margin-top:10px; }}

.adv-wrap{{ margin-top: 24px; }}
.adv-btn{{ display:inline-flex; align-items:center; gap:10px; padding:12px 16px; border-radius:12px; color:#052d14; font-weight:700;
          background: linear-gradient(90deg, #28d17a, #30d158 45%, #28d17a); border:none; cursor:pointer;
          box-shadow:0 8px 18px rgba(48,209,88,0.25), inset 0 1px 0 rgba(255,255,255,0.35); }}
.adv-card{{ margin-top:14px; display:none; }}
.adv-grid{{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
@media (max-width: 900px){{ .grid-2{{ grid-template-columns:1fr; }} .adv-grid{{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
  <div class="container">
    <div class="panel">

      <div class="header">
        <a class="home-btn-small top-home" href="/">🏠 Home</a>
        <div class="h-title">AI <span class="accent">Analysis</span> Complete</div>
        <div class="h-sub">Results for {out["crop"]} prediction in {out["state"]}, {out["district"]}</div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3>Summary</h3>
          <div class="kv"><span class="k">State</span><span class="v">{out["state"]}</span></div>
          <div class="kv"><span class="k">District</span><span class="v">{out["district"]}</span></div>
          <div class="kv"><span class="k">Crop</span><span class="v">{out["crop"]}</span></div>
          <div class="kv"><span class="k">Season</span><span class="v">{out["season"]}</span></div>
          <div class="kv"><span class="k">Area (hectares)</span><span class="v">{_f2(out["area_input"])}</span></div>
        </div>

        <div class="card">
          <h3>Predicted {out["crop"]} yield</h3>
          <div class="yield-big">{_f2(out["yield_per_hectare"])} tonnes/ha</div>
          <div class="yield-sub">Estimated total: {_f2(out["production_pred"])} tonnes</div>
        </div>
      </div>

      <!-- Growth Stage Card -->
      <div class="card progress-card">
        <h3>Growth Stage (as of {out["forecast_date"]})</h3>
        <div class="grid-2" style="margin-top:12px;">
          <div>
            <div class="kv"><span class="k">Sowing Date</span><span class="v">{out["sowing_date"]}</span></div>
            <div class="kv"><span class="k">Expected Harvest</span><span class="v">{out["harvest_date"]}</span></div>
            <div class="kv"><span class="k">Crop Duration</span><span class="v">{out["crop_duration_days"]} days</span></div>
          </div>
          <div>
            <div class="kv"><span class="k">Season Progress</span><span class="v">{out["season_progress_pct"]}%</span></div>
            <div class="kv"><span class="k">Days Elapsed</span><span class="v">{out["days_elapsed"]} days</span></div>
            <div class="kv"><span class="k">Days to Harvest</span><span class="v">{out["days_remaining"]} days</span></div>
            <div class="kv"><span class="k">Confidence</span><span class="v">{out["confidence_score"]}%</span></div>
          </div>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {out["season_progress_pct"]}%;"></div>
        </div>
      </div>

      <!-- Risk Assessment -->
      <div class="card" style="margin-top:18px;">
        <h3>Risk Assessment</h3>
        <div class="risk-badge" style="background:{risk_bg}; border:1px solid {risk_border}; color:{risk_color};">
          Risk Level: {risk_level}
        </div>
        <div style="margin-top:16px; padding:16px; background:rgba(0,0,0,0.2); border-radius:12px; white-space:pre-line; color:#d0e5da;">
{out['risk']['advisory']}
        </div>
      </div>

      <!-- Recommendation Cards -->
      <div class="grid-2" style="margin-top:18px;">
        <div class="card" style="background:linear-gradient(180deg, rgba(7,20,15,.6), rgba(7,20,15,.45)); border-color:rgba(48,209,88,0.18);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:44px;height:44px;border-radius:12px;background:rgba(48,209,88,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(48,209,88,.25);">💧</div>
            <h3 style="margin:0;">Irrigation Strategy</h3>
          </div>
          <div class="yield-big" style="margin-top:14px;">{irr_title}</div>
          <div class="yield-sub">{irr_sub}</div>
        </div>

        <div class="card" style="background:linear-gradient(180deg, rgba(36,28,5,.62), rgba(30,24,4,.45)); border-color:rgba(255,198,69,0.22);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:44px;height:44px;border-radius:12px;background:rgba(255,198,69,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,198,69,.3);">🧪</div>
            <h3 style="margin:0;">Fertilizer Blend</h3>
          </div>
          <div class="yield-big" style="margin-top:14px;">{fert_npk}</div>
          <div class="yield-sub">{fert_note}</div>
        </div>
      </div>

      <!-- Pesticides -->
      <div class="card" style="margin-top:18px;background:linear-gradient(180deg, rgba(18,7,32,.55), rgba(18,7,32,.45)); border-color:rgba(155,125,200,0.18);">
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="width:44px;height:44px;border-radius:12px;background:rgba(155,125,200,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(155,125,200,.3);">🛡️</div>
          <h3 style="margin:0;">Recommended Pesticides</h3>
        </div>
        <div class="pest-grid">{pest_grid_items}</div>
      </div>

      <!-- Advanced insights -->
      <div class="adv-wrap">
        <button class="adv-btn" id="advToggle" aria-expanded="false">Show advanced insights</button>
        <div class="card adv-card" id="advCard" aria-hidden="true" style="display:none;">
          <div class="adv-grid">
            <div class="card">
              <h3>Model Context</h3>
              <div class="kv"><span class="k">Latitude</span><span class="v">{_f3(out["lat"])}</span></div>
              <div class="kv"><span class="k">Longitude</span><span class="v">{_f3(out["lon"])}</span></div>
              <div class="kv"><span class="k">Forecast Date</span><span class="v">{out["forecast_date"]}</span></div>
            </div>
            <div class="card">
              <h3>Weather (Past 7 Days)</h3>
              <div class="kv"><span class="k">Rainfall_sum</span><span class="v">{_f2(out["weather"]["Rainfall_sum"])}</span></div>
              <div class="kv"><span class="k">Tavg_mean</span><span class="v">{_f2(out["weather"]["Tavg_mean"])}</span></div>
              <div class="kv"><span class="k">Tmax_mean</span><span class="v">{_f2(out["weather"]["Tmax_mean"])}</span></div>
              <div class="kv"><span class="k">Tmin_mean</span><span class="v">{_f2(out["weather"]["Tmin_mean"])}</span></div>
              <div class="kv"><span class="k">ET0_sum</span><span class="v">{_f2(out["weather"]["ET0_sum"])}</span></div>
              <div class="kv"><span class="k">GDD_sum</span><span class="v">{_f2(out["weather"]["GDD_sum"])}</span></div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
<script>
  document.addEventListener('DOMContentLoaded', function () {{
    var advBtn = document.getElementById('advToggle');
    var advCard = document.getElementById('advCard');
    if (!advBtn || !advCard) return;
    if (!advCard.style.display) advCard.style.display = 'none';
    advBtn.addEventListener('click', function () {{
      var show = advCard.style.display !== 'block';
      advCard.style.display = show ? 'block' : 'none';
      advBtn.textContent = show ? 'Hide advanced insights' : 'Show advanced insights';
      advCard.setAttribute('aria-hidden', show ? 'false' : 'true');
      advBtn.setAttribute('aria-expanded', show ? 'true' : 'false');
    }});
  }});
</script>
</body>
</html>
"""
        return HTMLResponse(html)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/predict/", response_class=HTMLResponse)
def predict_trailing(
    state: str = Form(...),
    district: str = Form(...),
    crop: str = Form(...),
    land_area: float = Form(...),
    sowing_date: str = Form(...),
    end_date: Optional[str] = Form(None)
):
    return predict(state, district, crop, land_area, sowing_date, end_date)
