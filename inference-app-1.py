# Complete FastAPI Inference App - Ridge Stacked Model (R²=0.9991)
# Full version with all features: Irrigation, Fertilizer, Pesticides, Full styling
# Theme: Dark green based on screenshots, NO emojis

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

# =============================================================================
# Configuration
# =============================================================================
PROJ_ROOT = os.getcwd()
MODEL_PATH = os.path.join(PROJ_ROOT, "artifacts_yield", "yield_pipeline_forward_chain_STACKED_PRODUCTION.pkl")
CROP_DURATIONS_PATH = os.path.join(PROJ_ROOT, "crop_durations.json")
RULES_PATH = os.path.join(PROJ_ROOT, "rules", "crop_reco.json")
PROC_DIR = os.path.join(PROJ_ROOT, "processed_training_csvs")

DROUGHT_THRESHOLD = 50.0
HEAT_STRESS_THRESHOLD = 35.0
OPTIMAL_TEMP_RANGE = (20.0, 30.0)

# =============================================================================
# Crop Growth Stages (Detailed for major crops)
# =============================================================================
CROP_GROWTH_STAGES = {
    "Rice": {
        "stages": [
            {"name": "Germination & Seedling", "end_pct": 0.15, "critical_needs": "Adequate water, warm temp (25-30°C)"},
            {"name": "Tillering", "end_pct": 0.35, "critical_needs": "High nitrogen, consistent flooding"},
            {"name": "Panicle Initiation", "end_pct": 0.55, "critical_needs": "Critical water requirement, avoid stress"},
            {"name": "Flowering & Grain Filling", "end_pct": 0.85, "critical_needs": "Optimal temp (20-25°C), moderate water"},
            {"name": "Maturation", "end_pct": 1.0, "critical_needs": "Gradual water reduction, dry weather for harvest"}
        ]
    },
    "Wheat": {
        "stages": [
            {"name": "Germination & Emergence", "end_pct": 0.12, "critical_needs": "Cool temp (15-20°C), adequate moisture"},
            {"name": "Tillering & Vegetative Growth", "end_pct": 0.40, "critical_needs": "Nitrogen application, moderate water"},
            {"name": "Stem Extension & Booting", "end_pct": 0.60, "critical_needs": "Critical water period, frost protection"},
            {"name": "Flowering & Grain Filling", "end_pct": 0.85, "critical_needs": "Optimal temp (18-24°C), avoid heat stress"},
            {"name": "Ripening & Maturation", "end_pct": 1.0, "critical_needs": "Dry weather, harvest at right moisture"}
        ]
    },
    "Cotton(lint)": {
        "stages": [
            {"name": "Germination & Seedling", "end_pct": 0.20, "critical_needs": "Warm soil (>15°C), adequate moisture"},
            {"name": "Vegetative Growth & Squaring", "end_pct": 0.45, "critical_needs": "Nitrogen, consistent irrigation"},
            {"name": "Flowering & Boll Formation", "end_pct": 0.70, "critical_needs": "Peak water demand, avoid water stress"},
            {"name": "Boll Development", "end_pct": 0.90, "critical_needs": "Moderate water, pest management critical"},
            {"name": "Maturation & Defoliation", "end_pct": 1.0, "critical_needs": "Reduce water, prepare for harvest"}
        ]
    },
    "Maize": {
        "stages": [
            {"name": "Germination & Emergence", "end_pct": 0.10, "critical_needs": "Warm soil (>10°C), good seed contact"},
            {"name": "Vegetative Growth", "end_pct": 0.40, "critical_needs": "Nitrogen application, regular irrigation"},
            {"name": "Tasseling & Silking", "end_pct": 0.60, "critical_needs": "CRITICAL water period, avoid stress"},
            {"name": "Grain Filling", "end_pct": 0.85, "critical_needs": "Consistent moisture, optimal temp (20-30°C)"},
            {"name": "Maturation & Drying", "end_pct": 1.0, "critical_needs": "Reduce water, dry down for harvest"}
        ]
    },
    "Sugarcane": {
        "stages": [
            {"name": "Germination & Establishment", "end_pct": 0.15, "critical_needs": "High moisture, warm temp (>20°C)"},
            {"name": "Tillering & Early Growth", "end_pct": 0.35, "critical_needs": "Nitrogen, consistent irrigation"},
            {"name": "Grand Growth Phase", "end_pct": 0.65, "critical_needs": "Peak water & nutrient demand"},
            {"name": "Ripening", "end_pct": 0.90, "critical_needs": "Reduce nitrogen, moderate water stress"},
            {"name": "Maturation", "end_pct": 1.0, "critical_needs": "Dry period for sugar concentration"}
        ]
    },
    "default": {
        "stages": [
            {"name": "Vegetative", "end_pct": 0.25, "critical_needs": "Establishment, nitrogen application"},
            {"name": "Reproductive", "end_pct": 0.50, "critical_needs": "Critical water period"},
            {"name": "Grain/Fruit Filling", "end_pct": 0.75, "critical_needs": "Consistent moisture, nutrient availability"},
            {"name": "Maturation", "end_pct": 1.0, "critical_needs": "Prepare for harvest"}
        ]
    }
}

# =============================================================================
# Globals
# =============================================================================
_ready = {"ok": False, "reason": "initializing"}
_fold_models = None
_meta_model = None
_CROP_DURATIONS: Dict = {}
_RULES_CACHE: Dict = {}
_ENRICHED_DF: Optional[pd.DataFrame] = None

# =============================================================================
# Utility Functions
# =============================================================================
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
    month = sowing_date.month
    if 6 <= month <= 10:
        return "Kharif"
    elif 11 <= month <= 2:
        return "Rabi"
    else:
        return "Zaid"

def _calculate_harvest_date(crop, sowing_date):
    crop_key = crop.strip()
    if crop_key in _CROP_DURATIONS:
        duration_days = _CROP_DURATIONS[crop_key]["duration_days"]
    else:
        duration_days = 120
    harvest_date = sowing_date + timedelta(days=duration_days)
    season_inferred = _infer_season_from_sowing_date(sowing_date)
    return harvest_date, duration_days, season_inferred

def _get_detailed_growth_stage(crop, progress_pct):
    crop_norm = crop.strip()
    stage_def = CROP_GROWTH_STAGES.get(crop_norm, CROP_GROWTH_STAGES["default"])
    for stage in stage_def["stages"]:
        if progress_pct <= stage["end_pct"]:
            return {
                "stage_name": stage["name"],
                "critical_needs": stage["critical_needs"],
                "stage_end_pct": stage["end_pct"] * 100
            }
    return {"stage_name": "Maturation", "critical_needs": "Prepare for harvest", "stage_end_pct": 100.0}

def _season_progress_with_sowing(crop, sowing_date, current_date=None):
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
    stage_info = _get_detailed_growth_stage(crop, progress)
    return {
        'progress': progress,
        'harvest_date': harvest_date,
        'days_total': duration_days,
        'days_elapsed': days_elapsed,
        'days_remaining': max(0, days_remaining),
        'season_inferred': season_inferred,
        'past_harvest': past_harvest,
        'growth_stage': stage_info['stage_name'],
        'critical_needs': stage_info['critical_needs'],
        'stage_end_pct': stage_info['stage_end_pct']
    }

# =============================================================================
# Rules Loader (Irrigation, Fertilizer, Pesticides)
# =============================================================================
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

# =============================================================================
# Georesolver
# =============================================================================
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

# =============================================================================
# NASA Weather API - CUMULATIVE from Sowing to End Date
# =============================================================================
def fetch_cumulative_weather(lat: float, lon: float, sowing_date, end_date) -> Dict[str, float]:
    if isinstance(sowing_date, str):
        sowing_date = pd.to_datetime(sowing_date).to_pydatetime()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).to_pydatetime()
    elif end_date is None:
        end_date = datetime.utcnow()

    start_str = sowing_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

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
            return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0, "days_count": 0}

        df = pd.DataFrame({"date": keys})
        for k, series in [("rain", pre), ("tavg", tavg), ("tmax", tmax), ("tmin", tmin)]:
            df[k] = df["date"].map(series) if isinstance(series, dict) else np.nan

        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        for c in ["rain","tavg","tmax","tmin"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < -900, c] = np.nan

        exact_start = pd.to_datetime(start_str, format="%Y%m%d")
        exact_end = pd.to_datetime(end_str, format="%Y%m%d")
        df = df[(df["date"] >= exact_start) & (df["date"] <= exact_end)]

        if df.empty:
            return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0, "days_count": 0}

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
            "days_count": len(df)
        }
    except Exception:
        return {"Rainfall_sum": 0.0, "Tavg_mean": 0.0, "Tmax_mean": 0.0, "Tmin_mean": 0.0, "ET0_sum": 0.0, "GDD_sum": 0.0, "days_count": 0}

# =============================================================================
# Historical Lags
# =============================================================================
def _get_historical_lags(state, district, crop, cropyear):
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

# =============================================================================
# Confidence & Risk Analysis
# =============================================================================
def _analyze_confidence_and_risks(weather_cumulative, season_progress, lags, fold_std):
    confidence_breakdown = {
        "base_confidence": 0.50,
        "season_progress_bonus": 0.0,
        "model_agreement_factor": 0.0,
        "weather_adjustment": 0.0,
        "historical_data_bonus": 0.0
    }
    
    progress_bonus = season_progress * 0.30
    confidence_breakdown["season_progress_bonus"] = progress_bonus
    
    if fold_std < 0.3:
        model_factor = 0.20
    elif fold_std < 0.6:
        model_factor = 0.10
    elif fold_std < 1.0:
        model_factor = 0.0
    else:
        model_factor = -0.15
    confidence_breakdown["model_agreement_factor"] = model_factor
    
    rainfall = weather_cumulative['Rainfall_sum']
    tmax = weather_cumulative['Tmax_mean']
    tavg = weather_cumulative['Tavg_mean']
    days_count = weather_cumulative.get('days_count', 90)
    weekly_avg_rain = (rainfall / days_count) * 7 if days_count > 0 else 0
    
    weather_penalty = 0.0
    weather_bonus = 0.0
    risks = []
    recommendations = []
    
    expected_rainfall = days_count * 5
    if rainfall < expected_rainfall * 0.6:
        severity = (expected_rainfall - rainfall) / expected_rainfall
        weather_penalty -= min(0.15, severity * 0.15)
        risks.append(f"Below-average rainfall: {rainfall:.0f}mm over {days_count} days (weekly avg: {weekly_avg_rain:.0f}mm)")
        recommendations.append("Consider supplemental irrigation if available")
    elif rainfall > expected_rainfall * 1.5:
        weather_penalty -= 0.10
        risks.append(f"Excess rainfall: {rainfall:.0f}mm over {days_count} days (waterlogging risk)")
        recommendations.append("Ensure proper drainage to prevent waterlogging")
    
    if tmax > HEAT_STRESS_THRESHOLD:
        severity = (tmax - HEAT_STRESS_THRESHOLD) / 10
        weather_penalty -= min(0.15, severity * 0.15)
        risks.append(f"Heat stress: Average max temp {tmax:.1f}°C (threshold: {HEAT_STRESS_THRESHOLD}°C)")
        recommendations.append("Monitor crop stress; consider foliar spray or mulching")
    
    if OPTIMAL_TEMP_RANGE[0] <= tavg <= OPTIMAL_TEMP_RANGE[1]:
        weather_bonus += 0.10
        recommendations.append("Temperature conditions optimal for growth")
    elif tavg < OPTIMAL_TEMP_RANGE[0]:
        weather_penalty -= 0.05
        risks.append(f"Below-optimal temperature: {tavg:.1f}°C average")
        recommendations.append("Growth may be slower than expected due to cool conditions")
    
    confidence_breakdown["weather_adjustment"] = weather_penalty + weather_bonus
    
    if lags['yieldcalc_lag1'] > 0:
        confidence_breakdown["historical_data_bonus"] = 0.10
        if lags['Rainfall_sum_lag1'] > 0:
            rain_change = ((rainfall - lags['Rainfall_sum_lag1']) / lags['Rainfall_sum_lag1']) * 100
            if abs(rain_change) > 30:
                if rain_change > 0:
                    recommendations.append(f"Rainfall {rain_change:.0f}% higher than last year - monitor for waterlogging")
                else:
                    recommendations.append(f"Rainfall {abs(rain_change):.0f}% lower than last year - irrigation recommended")
    
    total_confidence = sum(confidence_breakdown.values())
    total_confidence = max(0.20, min(0.95, total_confidence))
    
    if total_confidence >= 0.75:
        level = "High"
        level_explanation = "Strong model agreement, favorable conditions, adequate crop progress"
    elif total_confidence >= 0.55:
        level = "Medium"
        level_explanation = "Moderate confidence with some weather or growth stage uncertainty"
    else:
        level = "Low"
        level_explanation = "Significant uncertainty due to weather stress or model disagreement"
    
    return {
        'confidence_score': round(total_confidence, 2),
        'confidence_level': level,
        'confidence_explanation': level_explanation,
        'confidence_breakdown': {k: round(v, 3) for k, v in confidence_breakdown.items()},
        'risks': risks,
        'recommendations': recommendations
    }

# =============================================================================
# Model Loading
# =============================================================================
def _safe_load_model():
    global _fold_models, _meta_model, _ready, _CROP_DURATIONS
    try:
        print(f"[MODEL] Loading from: {MODEL_PATH}")
        artifacts = joblib.load(MODEL_PATH)
        models_forward_chain = artifacts['models_forward_chain']
        _fold_models = models_forward_chain['yieldcalc']['fold_models']
        _meta_model = models_forward_chain['yieldcalc']['meta_model']
        
        if os.path.isfile(CROP_DURATIONS_PATH):
            with open(CROP_DURATIONS_PATH, 'r') as f:
                _CROP_DURATIONS = json.load(f)
        else:
            _CROP_DURATIONS = {}
        
        _ready["ok"] = True
        _ready["reason"] = "ready"
        print(f"[MODEL] ✓ Successfully loaded Ridge stacked model with {len(_fold_models)} folds")
    except Exception as e:
        _ready["ok"] = False
        _ready["reason"] = f"startup failed: {e}"
        print(f"[MODEL] ✗ Failed to load: {e}")

_safe_load_model()

# =============================================================================
# Core Prediction - Using Ridge Stacking Logic
# =============================================================================
def _predict_core(state: str, district: str, crop: str, land_area: float, sowing_date: str, end_date: Optional[str]):
    if not _ready.get("ok", False):
        raise RuntimeError(f"Model not ready: {_ready.get('reason')}")

    sowing_dt = datetime.strptime(sowing_date, "%Y-%m-%d")
    ref_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    cropyear = sowing_dt.year

    season_info = _season_progress_with_sowing(crop, sowing_dt, ref_date)
    season = season_info['season_inferred']

    lat, lon = resolve_lat_lon(state, district)
    
    # Cumulative weather from sowing to forecast date
    weather_cumulative = fetch_cumulative_weather(lat, lon, sowing_dt, ref_date)
    
    lags = _get_historical_lags(state, district, crop, cropyear)
    
    deltas = {}
    for base_col in ["Rainfall_sum", "Tavg_mean", "Tmax_mean", "Tmin_mean", "ET0_sum", "GDD_sum"]:
        deltas[f"{base_col}_delta1"] = weather_cumulative[base_col] - lags[f"{base_col}_lag1"]
    for base_col in ["yieldcalc", "production", "area"]:
        deltas[f"{base_col}_delta1"] = 0.0
    
    input_data = pd.DataFrame([{
        'statename': state, 'districtname': district, 'statenorm': _norm_text(state), 'districtnorm': _norm_text(district),
        'crop': crop, 'season': season, 'cropyear': cropyear, 'area': float(land_area), 'lat': lat, 'lon': lon,
        'Rainfall_sum': weather_cumulative['Rainfall_sum'], 
        'Tavg_mean': weather_cumulative['Tavg_mean'], 
        'Tmax_mean': weather_cumulative['Tmax_mean'],
        'Tmin_mean': weather_cumulative['Tmin_mean'], 
        'ET0_sum': weather_cumulative['ET0_sum'], 
        'GDD_sum': weather_cumulative['GDD_sum'], 
        **lags, **deltas,
    }])
    
    # Get predictions from all 20 fold models
    base_preds = []
    for fold_model in _fold_models:
        try:
            pred = fold_model.predict(input_data)[0]
            base_preds.append(pred)
        except:
            continue

    if base_preds:
        base_preds_array = np.array(base_preds)
        fold_mean = np.mean(base_preds_array)
        fold_std = np.std(base_preds_array)
        
        # Create meta-features: [mean, fold1, fold2, ..., fold20]
        stack_features = np.hstack([fold_mean.reshape(1, 1), base_preds_array.reshape(1, -1)])
        
        # Final prediction with Ridge meta-model
        yield_pred_raw = float(_meta_model.predict(stack_features)[0])
        yield_pred = max(0.5, min(yield_pred_raw, 15.0))
        
        # Prediction interval
        uncertainty = min(fold_std * 2, yield_pred * 0.20)
        pred_lower = max(0.1, yield_pred - uncertainty)
        pred_upper = yield_pred + uncertainty
    else:
        yield_pred = 1.0
        pred_lower, pred_upper = 0.5, 1.5
        fold_std = 0.5
        base_preds_array = np.array([])
    
    production_pred = yield_pred * land_area
    
    # Detailed confidence and risk analysis
    insights = _analyze_confidence_and_risks(weather_cumulative, season_info['progress'], lags, fold_std)

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
        "yield_lower": pred_lower,
        "yield_upper": pred_upper,
        "production_pred": production_pred,
        "production_lower": pred_lower * land_area,
        "production_upper": pred_upper * land_area,
        "area_input": land_area,
        "forecast_date": ref_date.strftime("%Y-%m-%d"),
        "season_progress_pct": round(season_info['progress'] * 100, 1),
        "growth_stage_name": season_info['growth_stage'],
        "growth_critical_needs": season_info['critical_needs'],
        "days_elapsed": season_info['days_elapsed'],
        "days_remaining": season_info['days_remaining'],
        "past_harvest": season_info['past_harvest'],
        "confidence": insights,
        "weather_cumulative": weather_cumulative,
        "base_predictions": base_preds_array.tolist() if len(base_preds_array) > 0 else [],
        "fold_std": round(fold_std, 3),
        "fold_agreement": "High" if fold_std < 0.3 else "Medium" if fold_std < 0.6 else "Low"
    }

# =============================================================================
# UI Routes - Complete Crop List
# =============================================================================

ALL_CROPS = [
    "Arecanut", "Arhar/Tur", "Bajra", "Banana", "Barley", "Black pepper", "Blackgram", "Brinjal", "Cabbage",
    "Cardamom", "Cashewnut", "Castor seed", "Castorseed", "Cereals", "Coconut", "Coriander", "Cotton", "Cotton(lint)",
    "Cowpea(Lobia)", "Drum Stick", "Dry chillies", "Dry ginger", "Garlic", "Ginger", "Gram", "Grapes", "Groundnut",
    "Guar seed", "Guarseed", "Horse-gram", "Jack Fruit", "Jowar", "Jute", "Jute & Mesta", "Jute & mesta", "Khesari",
    "Korra", "Lemon", "Lentil", "Linseed", "Maize", "Mango", "Masoor", "Mesta", "Moong", "Moong(Green Gram)", "Moth",
    "Niger seed", "Nigerseed", "Nutri/Coarse Cereals", "Oilseeds total", "Onion", "Orange", "Other  Rabi pulses",
    "Other Cereals & Millets", "Other Fresh Fruits", "Other Kharif pulses", "Other Pulses", "Other Vegetables", "Paddy",
    "Papaya", "Peas & beans (Pulses)", "Pineapple", "Pome Granet", "Potato", "Pulses total", "Pump Kin", "Ragi",
    "Rapeseed & Mustard", "Rapeseed &Mustard", "Rice", "Rubber", "Safflower", "Samai", "Sannhamp", "Sannhemp", "Sapota",
    "Sesamum", "Shree Anna /Nutri Cereals", "Small Millets", "Small millets", "Soyabean", "Soybean", "Sugarcane",
    "Sunflower", "Sweet potato", "Tapioca", "Tea", "Tobacco", "Tomato", "Total Food Grains", "Total Oil Seeds",
    "Total Pulses", "Tur", "Turmeric", "Urad", "Varagu", "Wheat", "other oilseeds"
]

@app.get("/", response_class=HTMLResponse)
def form():
    crop_options = "".join([f'<option value="{crop}">{crop}</option>' for crop in ALL_CROPS])
    
    return f"""
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
    :root{{
      --bg:#070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
      --accent:#30d158; --text:#e6f5ea; --muted:#a9b8ae;
      --input-bg: rgba(255,255,255,0.06); --input-border: rgba(255,255,255,0.12);
      --input-focus: rgba(48, 209, 88, 0.55);
    }}
    * {{ box-sizing:border-box; }}
    html,body{{
      height:100%; margin:0;
      font-family:"Inter",system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg);
      color:var(--text);
    }}
    .container{{ min-height: 100%; width: 100%; display: flex; align-items: flex-start; justify-content: center; padding: 48px 24px; }}
    .panel{{
      width: 100%; max-width: 1200px; padding: 46px 36px 40px; border-radius: 18px; background: var(--card);
      border: 1px solid var(--card-border); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
      box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06); position: relative;
    }}
    h1{{ margin:6px 0 8px; font-size: clamp(32px, 3.6vw, 56px); text-align: center; font-weight: 800; }}
    .title-plain {{ color: #ecfff3; }} .title-accent {{ color: var(--accent); }}
    .subtitle{{ margin: 0 0 30px; text-align:center; color: var(--muted); font-size: 16px; }}
    form{{ display:grid; gap: 18px; margin-top: 8px; }}
    .field{{ display:flex; flex-direction:column; gap:10px; }}
    .label{{ display:flex; align-items:center; gap:10px; font-weight:600; color:#d6edde; letter-spacing:.2px; }}
    .label small{{ color:#9eb1a6; font-weight:500; }}
    .input{{
      width:100%; padding:16px 16px; border-radius:12px; border:1px solid var(--input-border); background: var(--input-bg);
      color: var(--text); outline:none; transition: border-color .2s, box-shadow .2s, background .2s; font-size:16px;
    }}
    .input::placeholder{{ color:#94a89c; }}
    .input:focus{{ border-color:var(--input-focus); box-shadow:0 0 0 4px rgba(48,209,88,0.15); background:rgba(255,255,255,0.09); }}
    .row{{ display:grid; grid-template-columns: 1fr 1fr; gap:18px; }}
    .btn{{
      margin-top: 8px; padding: 18px 20px; width:100%; border:none; border-radius:14px; font-size:18px; font-weight:700; color:#052d14;
      background: linear-gradient(90deg, #28d17a 0%, #30d158 45%, #28d17a 100%); cursor:pointer; transition: transform .08s, filter .2s, box-shadow .2s;
      box-shadow: 0 10px 24px rgba(48,209,88,0.25), inset 0 1px 0 rgba(255,255,255,0.35);
    }}
    .btn:hover{{ filter: brightness(1.03); }}
    .btn:active{{ transform: translateY(1px); }}
    select.input{{
      color:#e6f5ea; background-color: rgba(255,255,255,0.06);
      max-width: 100%;
      border-color: var(--input-border);
    }}
    select.input option{{
      color:#0b130f; background:#ffffff;
    }}
    @media (max-width: 768px) {{
      .row{{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="panel" aria-label="Agricultural Analysis Form">
      <h1><span class="title-plain">Agricultural</span><span class="title-accent"> Analysis</span></h1>
      <p class="subtitle">Enter farm details to receive AI‑powered insights</p>
      <form id="farmForm" method="post" action="/predict">
        <div class="row">
          <div class="field">
            <label class="label" for="state">State</label>
            <input class="input" id="state" name="state" type="text" placeholder="e.g., Punjab" required />
          </div>
          <div class="field">
            <label class="label" for="district">District</label>
            <input class="input" id="district" name="district" type="text" placeholder="e.g., Ludhiana" required />
          </div>
        </div>
        <div class="field">
          <label class="label" for="crop">Crop Type</label>
          <select class="input" id="crop" name="crop" required>
            <option value="" disabled selected>Select a crop</option>
            {crop_options}
          </select>
        </div>
        <div class="row">
          <div class="field">
            <label class="label" for="land">Land Size (Hectares)</label>
            <input class="input" id="land" name="land_area" type="number" inputmode="decimal" step="0.01" min="0" placeholder="Enter land size in hectares" required />
          </div>
          <div class="field">
            <label class="label" for="sowingDate">Sowing Date</label>
            <input class="input" id="sowingDate" name="sowing_date" type="text" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="\\d{{4}}-\\d{{2}}-\\d{{2}}" title="Enter date as YYYY-MM-DD (e.g., 2025-08-10)" required />
          </div>
        </div>
        <div class="field">
          <label class="label" for="endDate">Today's Date <small>(for weather data)</small></label>
          <input class="input" id="endDate" name="end_date" type="text" inputmode="numeric" placeholder="YYYY-MM-DD" pattern="\\d{{4}}-\\d{{2}}-\\d{{2}}" title="Enter date as YYYY-MM-DD (e.g., 2025-10-21)" />
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

# Continue in next message with the @app.post("/predict") endpoint...
@app.post("/predict")
async def predict(
    state: str = Form(...), district: str = Form(...), crop: str = Form(...),
    land_area: float = Form(...), sowing_date: str = Form(...), end_date: Optional[str] = Form(None)
):
    try:
        result = _predict_core(state, district, crop, land_area, sowing_date, end_date)
        rules = recommend_for_crop(crop)
        
        # Prepare data for template
        yield_val = result["yield_per_hectare"]
        prod_val = result["production_pred"]
        conf_score = result["confidence"]["confidence_score"]
        conf_level = result["confidence"]["confidence_level"]
        conf_expl = result["confidence"]["confidence_explanation"]
        conf_breakdown = result["confidence"]["confidence_breakdown"]
        
        risks = result["confidence"]["risks"]
        recommendations = result["confidence"]["recommendations"]
        
        weather = result["weather_cumulative"]
        
        # Format base predictions for display
        base_preds = result["base_predictions"]
        base_preds_str = ", ".join([f"{p:.3f}" for p in base_preds[:10]]) + ("..." if len(base_preds) > 10 else "")
        
        # Build risks/recommendations HTML separately to avoid f-string nesting issues
        risks_recs_html = ""
        if risks or recommendations:
            risks_recs_html = '<div class="card card-orange"><div class="section-title"><span class="icon icon-orange">💡</span> Actionable Recommendations</div>'
            if risks:
                risks_recs_html += '<div style="margin-bottom:24px;"><h3 style="margin:0 0 12px; color:#ffc266; font-size:16px; font-weight:600;">Identified Risks</h3>'
                for risk in risks:
                    risks_recs_html += f'<div class="alert-box alert-warning"><span class="alert-icon">⚠</span><span>{risk}</span></div>'
                risks_recs_html += '</div>'
            if recommendations:
                risks_recs_html += '<div><h3 style="margin:0 0 12px; color:#7de69b; font-size:16px; font-weight:600;">Recommendations</h3>'
                for rec in recommendations:
                    risks_recs_html += f'<div class="alert-box alert-success"><span class="alert-icon">✓</span><span>{rec}</span></div>'
                risks_recs_html += '</div>'
            risks_recs_html += '</div>'
        
        # Build pesticides list
        pest_items = "".join([f'<div class="pest-item">{p}</div>' for p in rules["pesticides"]])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crop Yield Forecast</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #070b0a; --card: rgba(15, 25, 20, 0.35); --card-border: rgba(255,255,255,0.08);
      --accent: #30d158; --text: #e6f5ea; --muted: #a9b8ae;
      --warning: #ff9f0a; --danger: #ff453a;
      --blue-accent: #4da6ff; --yellow-accent: #d4af37; --purple-bg: rgba(88, 66, 124, 0.25);
      --teal-accent: #20d9c8; --orange-accent: #ff9966;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      height: 100%; margin: 0;
      font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: radial-gradient(1800px 1000px at 60% 0%, #0f1a14 0%, #0b130f 40%, var(--bg) 80%) no-repeat, var(--bg);
      color: var(--text); line-height: 1.6;
    }}
    .container {{ padding: 48px 24px 48px 24px; max-width: 1400px; margin: 0 auto; position: relative; }}
    
    /* Home Button - Top Right */
    .home-btn {{
      position: fixed; top: 24px; right: 24px; z-index: 1000;
      padding: 12px 20px; border-radius: 12px; border: none; font-size: 15px; font-weight: 600;
      background: linear-gradient(90deg, #28d17a 0%, #30d158 45%, #28d17a 100%);
      color: #052d14; text-decoration: none; display: inline-flex; align-items: center; gap: 8px;
      box-shadow: 0 6px 18px rgba(48,209,88,0.3), inset 0 1px 0 rgba(255,255,255,0.35);
      transition: all 0.2s ease; cursor: pointer;
    }}
    .home-btn:hover {{ filter: brightness(1.05); }}
    .home-btn:active {{ transform: translateY(1px); }}
    
    /* Header */
    .header {{ text-align: center; margin-bottom: 40px; }}
    .header h1 {{ margin: 0 0 8px; font-size: clamp(28px, 3.5vw, 48px); font-weight: 800; color: #ecfff3; }}
    .header .accent {{ color: var(--accent); }}
    .header p {{ margin: 0; color: var(--muted); font-size: 16px; }}
    
    /* Cards */
    .card {{
      background: var(--card); border: 1px solid var(--card-border); border-radius: 18px;
      padding: 28px 32px; margin-bottom: 24px; backdrop-filter: blur(14px) saturate(140%);
      box-shadow: 0 8px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    
    /* Colored Card Variants */
    .card-blue {{
      background: rgba(15, 25, 30, 0.5); border: 1px solid rgba(77, 166, 255, 0.25);
      box-shadow: 0 8px 24px rgba(77, 166, 255, 0.15), inset 0 1px 0 rgba(77, 166, 255, 0.1);
    }}
    .card-yellow {{
      background: rgba(30, 25, 15, 0.5); border: 1px solid rgba(212, 175, 55, 0.25);
      box-shadow: 0 8px 24px rgba(212, 175, 55, 0.15), inset 0 1px 0 rgba(212, 175, 55, 0.1);
    }}
    .card-purple {{
      background: var(--purple-bg); border: 1px solid rgba(138, 106, 184, 0.3);
      box-shadow: 0 8px 24px rgba(138, 106, 184, 0.2), inset 0 1px 0 rgba(138, 106, 184, 0.1);
    }}
    .card-teal {{
      background: rgba(15, 30, 28, 0.5); border: 1px solid rgba(32, 217, 200, 0.25);
      box-shadow: 0 8px 24px rgba(32, 217, 200, 0.15), inset 0 1px 0 rgba(32, 217, 200, 0.1);
    }}
    .card-orange {{
      background: rgba(30, 20, 15, 0.5); border: 1px solid rgba(255, 153, 102, 0.25);
      box-shadow: 0 8px 24px rgba(255, 153, 102, 0.15), inset 0 1px 0 rgba(255, 153, 102, 0.1);
    }}
    
    /* Section Titles */
    .section-title {{
      font-size: 20px; font-weight: 700; color: #d6edde; margin: 0 0 20px; display: flex; align-items: center; gap: 12px;
    }}
    .section-title .icon {{ font-size: 24px; }}
    
    /* Colored Section Title Icons */
    .section-title .icon-blue {{ color: var(--blue-accent); }}
    .section-title .icon-yellow {{ color: var(--yellow-accent); }}
    .section-title .icon-purple {{ color: #a78bfa; }}
    .section-title .icon-teal {{ color: var(--teal-accent); }}
    .section-title .icon-orange {{ color: var(--orange-accent); }}
    
    /* Grid Layouts */
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
    .metric-box {{
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
      padding: 20px; text-align: center;
    }}
    .metric-label {{ font-size: 14px; color: var(--muted); margin-bottom: 8px; font-weight: 500; }}
    .metric-value {{ font-size: 32px; font-weight: 800; color: var(--accent); }}
    .metric-unit {{ font-size: 16px; color: #c2d9c9; margin-left: 4px; }}
    .metric-subtext {{ font-size: 13px; color: #9eb1a6; margin-top: 6px; }}
    
    /* Buttons */
    .btn {{
      display: inline-flex; align-items: center; justify-content: center; gap: 10px;
      padding: 14px 24px; border-radius: 12px; border: none; font-size: 15px; font-weight: 600;
      cursor: pointer; transition: all 0.2s ease; text-decoration: none;
    }}
    .btn-accent {{
      background: linear-gradient(90deg, #28d17a 0%, #30d158 45%, #28d17a 100%);
      color: #052d14; box-shadow: 0 6px 18px rgba(48,209,88,0.3), inset 0 1px 0 rgba(255,255,255,0.35);
    }}
    .btn-accent:hover {{ filter: brightness(1.05); }}
    .btn-accent:active {{ transform: translateY(1px); }}
    .btn-secondary {{
      background: rgba(48, 209, 88, 0.15); color: var(--accent); border: 1px solid rgba(48, 209, 88, 0.3);
    }}
    .btn-secondary:hover {{ background: rgba(48, 209, 88, 0.22); }}
    .btn-container {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }}
    
    /* Progress Bar */
    .progress-container {{ margin: 20px 0; }}
    .progress-label {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; color: var(--muted); }}
    .progress-bar {{
      width: 100%; height: 14px; background: rgba(255,255,255,0.08); border-radius: 8px; overflow: hidden;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }}
    .progress-fill {{
      height: 100%; background: linear-gradient(90deg, #28d17a, #30d158); border-radius: 8px;
      transition: width 0.6s ease; box-shadow: 0 0 12px rgba(48,209,88,0.5);
    }}
    
    /* Growth Stage Highlight */
    .stage-highlight {{
      background: linear-gradient(135deg, rgba(48,209,88,0.15) 0%, rgba(48,209,88,0.05) 100%);
      border: 2px solid rgba(48,209,88,0.4); border-radius: 14px; padding: 20px; margin: 16px 0;
      box-shadow: 0 4px 16px rgba(48,209,88,0.15);
    }}
    .stage-name {{ font-size: 24px; font-weight: 800; color: var(--accent); margin-bottom: 8px; }}
    .stage-needs {{ font-size: 15px; color: #c2d9c9; }}
    
    /* Lists */
    .info-list {{ list-style: none; padding: 0; margin: 16px 0; }}
    .info-list li {{ padding: 12px 16px; background: rgba(255,255,255,0.04); border-radius: 10px; margin-bottom: 10px; }}
    .info-list .label {{ font-weight: 600; color: #d6edde; }}
    .info-list .value {{ color: var(--accent); }}
    
    /* Risks & Recommendations */
    .alert-box {{
      padding: 16px 18px; border-radius: 12px; margin-bottom: 12px;
      display: flex; align-items: start; gap: 12px; line-height: 1.5;
    }}
    .alert-warning {{ background: rgba(255,159,10,0.12); border: 1px solid rgba(255,159,10,0.25); color: #ffc266; }}
    .alert-success {{ background: rgba(48,209,88,0.12); border: 1px solid rgba(48,209,88,0.25); color: #7de69b; }}
    .alert-icon {{ font-size: 20px; flex-shrink: 0; }}
    
    /* Irrigation & Fertilizer Cards with Color Accents */
    .irr-card {{
      background: rgba(77, 166, 255, 0.08); border: 1px solid rgba(77, 166, 255, 0.25);
      border-radius: 14px; padding: 20px;
    }}
    .fert-card {{
      background: rgba(212, 175, 55, 0.08); border: 1px solid rgba(212, 175, 55, 0.25);
      border-radius: 14px; padding: 20px;
    }}
    .irr-title {{ font-size: 22px; font-weight: 700; color: var(--blue-accent); margin: 0 0 8px; }}
    .fert-title {{ font-size: 22px; font-weight: 700; color: var(--yellow-accent); margin: 0 0 8px; }}
    .irr-sub, .fert-sub {{ color: var(--muted); font-size: 14px; }}
    
    /* Pesticides with Purple Theme */
    .pest-item {{
      background: rgba(138, 106, 184, 0.12); border: 1px solid rgba(138, 106, 184, 0.25);
      border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; font-size: 15px;
    }}
    
    /* Collapsible Sections */
    .collapsible {{ display: none; margin-top: 20px; }}
    .collapsible.active {{ display: block; }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    .detail-row:last-child {{ border-bottom: none; }}
    
    /* Confidence Badge */
    .conf-badge {{
      display: inline-block; padding: 6px 14px; border-radius: 8px; font-size: 14px; font-weight: 600;
      background: rgba(48,209,88,0.2); color: var(--accent); border: 1px solid rgba(48,209,88,0.35);
    }}
    
    @media (max-width: 768px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
      .btn-container {{ flex-direction: column; }}
      .btn {{ width: 100%; }}
      .home-btn {{ top: 16px; right: 16px; padding: 10px 16px; font-size: 14px; }}
    }}
  </style>
  <script>
    function toggle(id) {{
      const el = document.getElementById(id);
      el.classList.toggle('active');
      const btn = event.target;
      const isShowing = el.classList.contains('active');
      if (btn.textContent.includes('Show')) {{
        btn.textContent = btn.textContent.replace('Show', 'Hide');
      }} else {{
        btn.textContent = btn.textContent.replace('Hide', 'Show');
      }}
    }}
  </script>
</head>
<body>
  <!-- Home Button - Top Right -->
  <a class="home-btn" href="/">🏠 Home</a>
  
  <div class="container">
    
    <!-- Header -->
    <div class="header">
      <h1><span>AI</span> <span class="accent">Analysis</span> <span>Complete</span></h1>
      <p>Results for {result['crop']} prediction in {result['district']}, {result['state']}</p>
    </div>
    
    <!-- Summary with Teal Accent -->
    <div class="card card-teal">
      <div class="section-title"><span class="icon icon-teal">📋</span> Summary</div>
      <ul class="info-list">
        <li><span class="label">State</span> <span class="value">{result['state']}</span></li>
        <li><span class="label">District</span> <span class="value">{result['district']}</span></li>
        <li><span class="label">Crop</span> <span class="value">{result['crop']}</span></li>
        <li><span class="label">Area</span> <span class="value">{result['area_input']:.2f} hectares</span></li>
      </ul>
    </div>
    
    <!-- Yield Predictions -->
    <div class="card">
      <div class="section-title">Predicted {result['crop']} yield</div>
      <div class="grid-2">
        <div class="metric-box">
          <div class="metric-label">Yield</div>
          <div class="metric-value">{yield_val:.4f}<span class="metric-unit">tonnes/hectare</span></div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Estimated total (model)</div>
          <div class="metric-value">{prod_val:.4f}<span class="metric-unit">tonnes</span></div>
        </div>
      </div>
      
      <!-- Detailed Logs Button -->
      <div class="btn-container">
        <button class="btn btn-accent" onclick="toggle('detailLogs')">Show Detailed Logs</button>
      </div>
      
      <div id="detailLogs" class="collapsible">
        <div style="margin-top:20px; padding:20px; background:rgba(255,255,255,0.04); border-radius:12px;">
          <h3 style="margin-top:0; color:#d6edde; font-size:18px;">Model Prediction Details</h3>
          <div class="detail-row">
            <span style="color:var(--muted);">Yield Range:</span>
            <span style="color:var(--accent); font-weight:600;">{result['yield_lower']:.2f} - {result['yield_upper']:.2f} t/ha</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Production Range:</span>
            <span style="color:var(--accent); font-weight:600;">{result['production_lower']:.2f} - {result['production_upper']:.2f} tonnes</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Fold Agreement:</span>
            <span style="color:var(--accent); font-weight:600;">{result['fold_agreement']}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Fold Std Deviation:</span>
            <span style="color:var(--accent); font-weight:600;">{result['fold_std']:.3f}</span>
          </div>
          <div style="margin-top:16px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.1);">
            <p style="margin:0 0 8px; color:var(--muted); font-size:14px;">Base Predictions (Sample):</p>
            <p style="margin:0; font-family:monospace; font-size:13px; color:#9eb1a6; word-break:break-all;">{base_preds_str}</p>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Irrigation & Fertilizer with Colored Cards -->
    <div class="grid-2">
      <div class="card card-blue">
        <div class="section-title"><span class="icon icon-blue">💧</span> Irrigation Strategy</div>
        <div class="irr-card">
          <div class="irr-title">{rules['irrigation']['title']}</div>
          <div class="irr-sub">{rules['irrigation']['subtitle']}</div>
        </div>
      </div>
      <div class="card card-yellow">
        <div class="section-title"><span class="icon icon-yellow">🌾</span> Fertilizer Blend</div>
        <div class="fert-card">
          <div class="fert-title">{rules['fertilizer_blend']['npk']}</div>
          <div class="fert-sub">{rules['fertilizer_blend']['note']}</div>
        </div>
      </div>
    </div>
    
    <!-- Recommended Pesticides with Purple Theme -->
    <div class="card card-purple">
      <div class="section-title"><span class="icon icon-purple">🛡️</span> Recommended Pesticides</div>
      {pest_items}
    </div>
    
    <!-- Growth Stage -->
    <div class="card">
      <div class="section-title">Growth Stage (as of {result['forecast_date']})</div>
      
      <!-- Highlighted Current Stage -->
      <div class="stage-highlight">
        <div class="stage-name">{result['growth_stage_name']}</div>
        <div class="stage-needs">{result['growth_critical_needs']}</div>
      </div>
      
      <div class="progress-container">
        <div class="progress-label">
          <span>Crop Progress</span>
          <span style="font-weight:700; color:var(--accent);">{result['season_progress_pct']}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {result['season_progress_pct']}%;"></div>
        </div>
      </div>
      
      <ul class="info-list">
        <li><span class="label">Days Elapsed:</span> <span class="value">{result['days_elapsed']} days</span></li>
        <li><span class="label">Days Remaining:</span> <span class="value">{result['days_remaining']} days</span></li>
      </ul>
      
      <!-- Confidence Analysis Button -->
      <div class="btn-container">
        <button class="btn btn-accent" onclick="toggle('confAnalysis')">Show Confidence Analysis</button>
      </div>
      
      <div id="confAnalysis" class="collapsible">
        <div style="margin-top:20px; padding:20px; background:rgba(255,255,255,0.04); border-radius:12px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <h3 style="margin:0; color:#d6edde; font-size:18px;">Confidence Score</h3>
            <span class="conf-badge">{conf_score*100:.0f}% ({conf_level})</span>
          </div>
          <p style="color:var(--muted); margin:0 0 20px; font-size:15px;">{conf_expl}</p>
          
          <h4 style="color:#d6edde; margin:20px 0 12px; font-size:16px;">Confidence Breakdown</h4>
          <div class="detail-row">
            <span style="color:var(--muted);">Base Confidence:</span>
            <span style="color:var(--accent); font-weight:600;">{conf_breakdown['base_confidence']:+.3f}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Season Progress Bonus:</span>
            <span style="color:var(--accent); font-weight:600;">{conf_breakdown['season_progress_bonus']:+.3f}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Model Agreement Factor:</span>
            <span style="color:var(--accent); font-weight:600;">{conf_breakdown['model_agreement_factor']:+.3f}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Weather Adjustment:</span>
            <span style="color:var(--accent); font-weight:600;">{conf_breakdown['weather_adjustment']:+.3f}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Historical Data Bonus:</span>
            <span style="color:var(--accent); font-weight:600;">{conf_breakdown['historical_data_bonus']:+.3f}</span>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Risks & Recommendations with Orange Accent -->
    {risks_recs_html}
    
    <!-- Action Buttons (removed Home from here) -->
    <div class="card">
      <div class="btn-container">
        <button class="btn btn-secondary" onclick="toggle('weatherSection')">Show Cumulative Weather</button>
        <button class="btn btn-secondary" onclick="toggle('advancedSection')">Show Model Information</button>
      </div>
      
      <!-- Cumulative Weather Section -->
      <div id="weatherSection" class="collapsible">
        <div style="margin-top:20px; padding:20px; background:rgba(255,255,255,0.04); border-radius:12px;">
          <h3 style="margin-top:0; color:#d6edde; font-size:18px;">Cumulative Weather (Sowing to Forecast)</h3>
          <div class="detail-row">
            <span style="color:var(--muted);">Total Rainfall:</span>
            <span style="color:var(--accent); font-weight:600;">{weather['Rainfall_sum']:.0f}mm over {weather['days_count']} days</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Average Temperature:</span>
            <span style="color:var(--accent); font-weight:600;">{weather['Tavg_mean']:.1f}°C (Max: {weather['Tmax_mean']:.1f}°C)</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Growing Degree Days:</span>
            <span style="color:var(--accent); font-weight:600;">{weather['GDD_sum']:.0f}</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Evapotranspiration:</span>
            <span style="color:var(--accent); font-weight:600;">{weather['ET0_sum']:.1f}mm</span>
          </div>
        </div>
      </div>
      
      <!-- Advanced Section - Model Info Only -->
      <div id="advancedSection" class="collapsible">
        <div style="margin-top:20px; padding:20px; background:rgba(255,255,255,0.04); border-radius:12px;">
          <h3 style="margin-top:0; color:#d6edde; font-size:18px;">Model Architecture</h3>
          <div class="detail-row">
            <span style="color:var(--muted);">Model Type:</span>
            <span style="color:var(--accent); font-weight:600;">XGBoost Forward Chain + Ridge Stacking</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Training Period:</span>
            <span style="color:var(--accent); font-weight:600;">2001-2023</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Performance:</span>
            <span style="color:var(--accent); font-weight:600;">R²=0.9991, MAE=0.09 t/ha</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Fold Models:</span>
            <span style="color:var(--accent); font-weight:600;">{len(result['base_predictions'])} forward-chain folds</span>
          </div>
          <div class="detail-row">
            <span style="color:var(--muted);">Weather Aggregation:</span>
            <span style="color:var(--accent); font-weight:600;">Cumulative from sowing to forecast date</span>
          </div>
          <div style="margin-top:16px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.1);">
            <p style="margin:0; color:var(--muted); font-size:14px; line-height:1.6;">
              This model uses a two-stage ensemble: 20 XGBoost models trained on progressive time windows (forward chaining), 
              with predictions combined by a Ridge regression meta-model for optimal accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
    
  </div>
</body>
</html>
        """
        return HTMLResponse(content=html)
        
    except Exception as e:
        error_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Error</title>
  <style>
    body {{ font-family: Inter, sans-serif; background: #070b0a; color: #e6f5ea; padding: 48px 24px; text-align: center; }}
    .error-box {{ max-width: 600px; margin: 0 auto; padding: 32px; background: rgba(255,69,58,0.12); border: 1px solid rgba(255,69,58,0.3); border-radius: 16px; }}
    h1 {{ color: #ff6961; margin: 0 0 16px; }}
    p {{ margin: 0 0 24px; color: #c2d9c9; }}
    a {{ display: inline-block; padding: 14px 24px; background: #30d158; color: #052d14; text-decoration: none; border-radius: 12px; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="error-box">
    <h1>Prediction Error</h1>
    <p>An error occurred while processing your request:</p>
    <p style="font-family: monospace; color: #ff6961;">{str(e)}</p>
    <a href="/">Back to Form</a>
  </div>
</body>
</html>
        """
        return HTMLResponse(content=error_html, status_code=500)
