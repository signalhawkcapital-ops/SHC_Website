"""
Historical Calibration — Real Data Analysis
=============================================
Analyzes cached SPX/VIX data to compute:
- VIX-to-realized-vol ratio (VIX typically overstates actual moves)
- Day-of-week realized vol multipliers from actual data
- Monthly seasonality from actual data
- Regime-specific EM accuracy
- Historical EM hit rates (how often does SPX stay within 1σ?)

These stats are used to refine the strategy engine's predictions.
"""

import math
import os
import json
from datetime import date
from typing import Optional, Dict, List

import numpy as np

CALIBRATION_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "calibration.json"
)


def calibrate_from_bars(bars) -> dict:
    """
    Compute calibration stats from historical DailyBar data.
    
    Returns dict with:
    - vix_overstatement: ratio of VIX-implied EM to actual |move| (typically 1.2-1.6)
    - dow_multipliers: realized vol by day-of-week (0=Mon..4=Fri)
    - month_multipliers: realized vol by month (1=Jan..12=Dec)
    - regime_stats: EM accuracy per VIX regime
    - em_hit_rate_1sigma: % of days actual |move| < VIX-implied EM
    - avg_daily_range_pct: average (high-low)/open as percentage
    """
    if not bars or len(bars) < 30:
        return {"error": "Insufficient data", "count": len(bars) if bars else 0}

    # Compute daily stats
    records = []
    for b in bars:
        vix = b.vix_open
        spx = b.spx_open
        if vix <= 0 or spx <= 0:
            continue
        
        implied_em = spx * (vix / 100) / math.sqrt(252)
        actual_move = abs(b.spx_close - b.spx_open)
        day_range = b.spx_high - b.spx_low
        log_ret = math.log(b.spx_close / b.spx_open) if b.spx_close > 0 and b.spx_open > 0 else 0
        
        records.append({
            "date": b.trade_date,
            "dow": b.trade_date.weekday(),
            "month": b.trade_date.month,
            "vix": vix,
            "spx": spx,
            "implied_em": implied_em,
            "actual_move": actual_move,
            "day_range": day_range,
            "log_ret": log_ret,
            "within_1sigma": actual_move <= implied_em,
            "range_pct": day_range / spx * 100,
        })

    n = len(records)
    if n < 30:
        return {"error": "Insufficient valid records", "count": n}

    # ── VIX overstatement ratio ──
    implied = [r["implied_em"] for r in records]
    actual = [r["actual_move"] for r in records]
    vix_overstatement = float(np.mean(implied) / np.mean(actual)) if np.mean(actual) > 0 else 1.0

    # ── EM hit rate ──
    em_hit_rate = sum(1 for r in records if r["within_1sigma"]) / n * 100

    # ── Day-of-week realized vol ──
    dow_vols = {}
    overall_vol = float(np.std([r["log_ret"] for r in records]))
    for d in range(5):
        day_rets = [r["log_ret"] for r in records if r["dow"] == d]
        if len(day_rets) >= 10:
            day_vol = float(np.std(day_rets))
            dow_vols[d] = round(day_vol / overall_vol, 3) if overall_vol > 0 else 1.0
        else:
            dow_vols[d] = 1.0

    # ── Monthly seasonality ──
    month_vols = {}
    for m in range(1, 13):
        month_rets = [r["log_ret"] for r in records if r["month"] == m]
        if len(month_rets) >= 10:
            month_vol = float(np.std(month_rets))
            month_vols[m] = round(month_vol / overall_vol, 3) if overall_vol > 0 else 1.0
        else:
            month_vols[m] = 1.0

    # ── Regime-specific stats ──
    regime_stats = {}
    for regime_name, vix_low, vix_high in [
        ("low_vol", 0, 14), ("normal", 14, 20),
        ("elevated", 20, 30), ("high_vol", 30, 100)
    ]:
        regime_recs = [r for r in records if vix_low <= r["vix"] < vix_high]
        if len(regime_recs) >= 5:
            rim = [r["implied_em"] for r in regime_recs]
            ram = [r["actual_move"] for r in regime_recs]
            regime_stats[regime_name] = {
                "count": len(regime_recs),
                "avg_vix": round(float(np.mean([r["vix"] for r in regime_recs])), 1),
                "avg_implied_em": round(float(np.mean(rim)), 1),
                "avg_actual_move": round(float(np.mean(ram)), 1),
                "overstatement": round(float(np.mean(rim) / np.mean(ram)), 3) if np.mean(ram) > 0 else 1.0,
                "pct_within_1sigma": round(
                    sum(1 for r in regime_recs if r["within_1sigma"]) / len(regime_recs) * 100, 1),
                "avg_range_pct": round(float(np.mean([r["range_pct"] for r in regime_recs])), 3),
            }

    # ── Summary ──
    avg_range_pct = float(np.mean([r["range_pct"] for r in records]))
    annualized_vol = overall_vol * math.sqrt(252) * 100

    result = {
        "count": n,
        "date_range": f"{records[0]['date'].isoformat()} to {records[-1]['date'].isoformat()}",
        "vix_overstatement": round(vix_overstatement, 3),
        "em_hit_rate_1sigma": round(em_hit_rate, 1),
        "avg_daily_range_pct": round(avg_range_pct, 3),
        "annualized_realized_vol": round(annualized_vol, 1),
        "avg_vix": round(float(np.mean([r["vix"] for r in records])), 1),
        "vix_vs_realized": round(float(np.mean([r["vix"] for r in records])) - annualized_vol, 1),
        "dow_multipliers": dow_vols,
        "month_multipliers": month_vols,
        "regime_stats": regime_stats,
    }

    # Cache results
    try:
        os.makedirs(os.path.dirname(CALIBRATION_CACHE), exist_ok=True)
        # Convert date keys to strings for JSON
        serializable = result.copy()
        serializable["dow_multipliers"] = {str(k): v for k, v in dow_vols.items()}
        serializable["month_multipliers"] = {str(k): v for k, v in month_vols.items()}
        with open(CALIBRATION_CACHE, 'w') as f:
            json.dump(serializable, f, indent=2)
    except Exception:
        pass

    return result


def load_calibration() -> Optional[dict]:
    """Load cached calibration results."""
    if os.path.exists(CALIBRATION_CACHE):
        try:
            with open(CALIBRATION_CACHE, 'r') as f:
                data = json.load(f)
            # Convert string keys back to int
            if "dow_multipliers" in data:
                data["dow_multipliers"] = {int(k): v for k, v in data["dow_multipliers"].items()}
            if "month_multipliers" in data:
                data["month_multipliers"] = {int(k): v for k, v in data["month_multipliers"].items()}
            return data
        except Exception:
            pass
    return None


def get_calibrated_em(vix: float, spx: float, trade_date: date,
                       calibration: Optional[dict] = None) -> float:
    """
    Compute calibrated expected move using real historical data.
    
    The standard VIX formula (SPX * VIX/100 / √252) typically OVERSTATES
    actual moves by 20-60%. This function adjusts using the measured
    overstatement ratio from real data.
    """
    # Standard VIX-implied EM
    raw_em = spx * (vix / 100) / math.sqrt(252)

    if calibration is None:
        calibration = load_calibration()

    if calibration and "vix_overstatement" in calibration:
        # Adjust for VIX overstatement (VIX overpredicts realized vol)
        overstatement = calibration["vix_overstatement"]
        adjusted_em = raw_em / overstatement

        # Apply calibrated DoW multiplier
        dow_mults = calibration.get("dow_multipliers", {})
        dow_mult = dow_mults.get(trade_date.weekday(), 1.0)

        # Apply calibrated month multiplier
        month_mults = calibration.get("month_multipliers", {})
        month_mult = month_mults.get(trade_date.month, 1.0)

        return adjusted_em * dow_mult * month_mult
    else:
        # Fallback to standard formula with hardcoded adjustments
        from engine.strategy import DOW_VOL_MULTIPLIERS, MONTH_VOL_MULTIPLIERS
        dow_mult = DOW_VOL_MULTIPLIERS.get(trade_date.weekday(), 1.0)
        month_mult = MONTH_VOL_MULTIPLIERS.get(trade_date.month, 1.0)
        return raw_em * dow_mult * month_mult
