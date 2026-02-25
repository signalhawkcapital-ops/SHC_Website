"""
Batman Strategy Engine — Core Analytics
========================================
Statistical models for SPX 0DTE butterfly position sizing,
backtesting, and trap optimization.

Uses VIX, expected move, historical realized vol, and intraday
theta decay models to recommend butterfly widths, center gaps,
and optimal trap entry times.
"""

import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from datetime import datetime, date, time, timedelta
from enum import Enum


# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5  # 9:30–16:00 ET
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Historical day-of-week realized vol multipliers (SPX 2018-2025)
DOW_VOL_MULTIPLIERS = {
    0: 1.12,  # Monday — gap risk, weekend news
    1: 0.95,  # Tuesday — calmest day historically
    2: 1.05,  # Wednesday — FOMC days, mid-week rebalancing
    3: 0.92,  # Thursday — lowest realized vol
    4: 1.08,  # Friday — OpEx, weekly expiry positioning
}

# Historical monthly vol seasonality (relative multiplier)
MONTH_VOL_MULTIPLIERS = {
    1: 1.05, 2: 1.02, 3: 1.08, 4: 0.95, 5: 0.98, 6: 0.93,
    7: 0.90, 8: 1.02, 9: 1.12, 10: 1.15, 11: 1.00, 12: 0.92,
}


class Regime(Enum):
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH_VOL = "high_vol"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ────────────────────────────────────────────────────────────
# Regime parameters — calibrated from backtesting
# ────────────────────────────────────────────────────────────

@dataclass
class RegimeParams:
    """Parameters calibrated per volatility regime."""
    bfly_width_pct: float       # Width as fraction of expected move
    gap_pct: float              # Center gap as fraction of expected move
    trap_multiplier: float      # Trap width relative to bfly width
    base_confidence: int        # Base confidence score (0-100)
    pin_zone_width: float       # Fraction of bfly width for pin probability
    early_trap_threshold: float # SPX move threshold for early trap (fraction of EM)
    prime_trap_decay: float     # Fraction of theta decayed at prime window


REGIME_PARAMS: Dict[Regime, RegimeParams] = {
    Regime.LOW_VOL: RegimeParams(
        bfly_width_pct=0.55, gap_pct=0.08, trap_multiplier=0.85,
        base_confidence=82, pin_zone_width=0.35, early_trap_threshold=0.25,
        prime_trap_decay=0.40
    ),
    Regime.NORMAL: RegimeParams(
        bfly_width_pct=0.65, gap_pct=0.10, trap_multiplier=0.90,
        base_confidence=75, pin_zone_width=0.30, early_trap_threshold=0.30,
        prime_trap_decay=0.38
    ),
    Regime.ELEVATED: RegimeParams(
        bfly_width_pct=0.80, gap_pct=0.14, trap_multiplier=0.95,
        base_confidence=65, pin_zone_width=0.25, early_trap_threshold=0.35,
        prime_trap_decay=0.35
    ),
    Regime.HIGH_VOL: RegimeParams(
        bfly_width_pct=1.00, gap_pct=0.20, trap_multiplier=1.00,
        base_confidence=50, pin_zone_width=0.20, early_trap_threshold=0.40,
        prime_trap_decay=0.32
    ),
}

RISK_ADJUSTMENTS = {
    # CONSERVATIVE: Minimize capital at risk per trade
    # Narrower flies = lower debit = less money on the line
    # Higher trap trigger = only traps on confirmed moves (fewer but higher-quality)
    # Smaller trap = cheaper to add
    RiskProfile.CONSERVATIVE: {
        "em_width_frac": 0.50,    # Narrower = lower debit cost
        "trap_width_mult": 1.25,  # Smaller trap = cheaper
        "trigger_mult": 1.40,     # Waits for bigger move = fewer but safer traps
        "conf_bonus": 5,
        "label": "Lower debit, tighter flies, patient trap trigger",
    },
    # MODERATE: Backtested optimal balance
    RiskProfile.MODERATE: {
        "em_width_frac": 0.60,    # Backtested optimal
        "trap_width_mult": 1.50,  # Backtested optimal
        "trigger_mult": 1.00,     # Regime default triggers
        "conf_bonus": 0,
        "label": "Balanced — backtested optimal parameters",
    },
    # AGGRESSIVE: Maximize profit zone and capture rate
    # Wider flies = higher debit but much larger profit zone
    # Lower trap trigger = traps earlier = more frequent entries
    # Wider trap = bigger tent = more profit coverage
    RiskProfile.AGGRESSIVE: {
        "em_width_frac": 0.70,    # Wider = higher debit, bigger profit zone
        "trap_width_mult": 1.75,  # Wider trap tent
        "trigger_mult": 0.70,     # Traps earlier on smaller moves
        "conf_bonus": -5,
        "label": "Higher debit, wider profit zone, earlier traps",
    },
}


# ────────────────────────────────────────────────────────────
# Math helpers
# ────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes option price."""
    if T <= 1e-10:
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes theta (time decay per day)."""
    if T <= 1e-10:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    common = -(S * norm_pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == "call":
        return (common - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
    return (common + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365


def round_to_strike(value: float, increment: float = 5.0) -> float:
    """Round to nearest valid strike (default 5-point SPX strikes)."""
    return round(value / increment) * increment


# ────────────────────────────────────────────────────────────
# VIX / Expected Move
# ────────────────────────────────────────────────────────────

def vix_to_daily_em(vix: float, spx: float) -> float:
    """Convert VIX to daily expected move in points."""
    daily_sigma = vix / 100 / math.sqrt(TRADING_DAYS_PER_YEAR)
    return spx * daily_sigma


def classify_regime(vix: float) -> Regime:
    """Classify market volatility regime from VIX."""
    if vix < 14:
        return Regime.LOW_VOL
    elif vix < 20:
        return Regime.NORMAL
    elif vix < 30:
        return Regime.ELEVATED
    return Regime.HIGH_VOL


# ────────────────────────────────────────────────────────────
# Intraday Theta Model
# ────────────────────────────────────────────────────────────

def time_to_expiry_fraction(current_time: time) -> float:
    """
    Calculate fraction of trading day remaining.
    Returns value in [0, 1] where 1 = market open, 0 = market close.
    """
    open_minutes = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    close_minutes = MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute
    current_minutes = current_time.hour * 60 + current_time.minute

    total = close_minutes - open_minutes
    remaining = max(0, close_minutes - current_minutes)
    return remaining / total


def intraday_theta_rate(time_remaining_frac: float) -> float:
    """
    Relative theta burn rate. Theta accelerates as expiry approaches.
    Models the 1/sqrt(T) relationship for ATM 0DTE options.

    Returns a multiplier relative to the morning rate.
    """
    if time_remaining_frac <= 0.01:
        return 10.0  # cap
    # Normalized so that at open (frac=1.0) rate=1.0
    return 1.0 / math.sqrt(time_remaining_frac)


def theta_curve(steps: int = 100) -> List[Tuple[float, float, float]]:
    """
    Generate intraday theta decay curve.
    Returns: list of (fraction_elapsed, cumulative_decay, instantaneous_rate)
    """
    results = []
    cumulative = 0.0
    dt = 1.0 / steps
    # Integrate theta rate across the day
    total_integral = 0.0
    for i in range(steps):
        frac_remaining = 1.0 - (i + 0.5) / steps
        rate = intraday_theta_rate(frac_remaining)
        total_integral += rate * dt

    running = 0.0
    for i in range(steps):
        frac_elapsed = i / steps
        frac_remaining = 1.0 - (i + 0.5) / steps
        rate = intraday_theta_rate(frac_remaining)
        running += rate * dt
        cumulative = running / total_integral
        results.append((frac_elapsed, cumulative, rate))

    return results


# ────────────────────────────────────────────────────────────
# Butterfly P&L Calculator
# ────────────────────────────────────────────────────────────

@dataclass
class ButterflyPosition:
    """A single butterfly spread."""
    lower: float
    center: float
    upper: float
    option_type: str  # "call" or "put"
    quantity: int = 1
    debit_paid: float = 0.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def pnl_at_expiry(self, settlement: float) -> float:
        """P&L per contract at a given settlement price."""
        if self.option_type == "call":
            val = (max(0, settlement - self.lower)
                   - 2 * max(0, settlement - self.center)
                   + max(0, settlement - self.upper))
        else:
            val = (max(0, self.upper - settlement)
                   - 2 * max(0, self.center - settlement)
                   + max(0, self.lower - settlement))
        return (val - self.debit_paid) * self.quantity

    def pnl_before_expiry(self, spot: float, T: float, sigma: float,
                          r: float = 0.05) -> float:
        """Approximate P&L before expiry using Black-Scholes."""
        t = self.option_type
        val = (bs_price(spot, self.lower, T, r, sigma, t)
               - 2 * bs_price(spot, self.center, T, r, sigma, t)
               + bs_price(spot, self.upper, T, r, sigma, t))
        return (val - self.debit_paid) * self.quantity

    def max_profit(self) -> float:
        """Maximum profit at center strike."""
        return (self.center - self.lower) - self.debit_paid


@dataclass
class BatmanPosition:
    """Full batman = call butterfly + put butterfly."""
    call_fly: ButterflyPosition
    put_fly: ButterflyPosition

    def pnl_at_expiry(self, settlement: float) -> float:
        return self.call_fly.pnl_at_expiry(settlement) + self.put_fly.pnl_at_expiry(settlement)

    def pnl_before_expiry(self, spot: float, T: float, sigma: float,
                          r: float = 0.05) -> float:
        return (self.call_fly.pnl_before_expiry(spot, T, sigma, r)
                + self.put_fly.pnl_before_expiry(spot, T, sigma, r))

    @property
    def total_debit(self) -> float:
        return self.call_fly.debit_paid + self.put_fly.debit_paid

    @property
    def max_profit(self) -> float:
        return max(self.call_fly.max_profit(), self.put_fly.max_profit())


# ────────────────────────────────────────────────────────────
# Strategy Recommendation Engine
# ────────────────────────────────────────────────────────────

@dataclass
class StrategyRecommendation:
    """Output of the strategy engine."""
    # Inputs
    spx_open: float
    vix: float
    trade_date: str
    regime: str
    risk_profile: str
    risk_profile_desc: str

    # Core sizing
    butterfly_width: float
    center_gap: float
    start_offset: float
    call_center: float
    put_center: float

    # Full strikes
    call_lower: float
    call_upper: float
    put_lower: float
    put_upper: float

    # Statistics
    expected_move: float
    raw_expected_move: float  # VIX-implied EM (used for width sizing)
    daily_sigma_pct: float
    range_1sigma: Tuple[float, float]
    range_2sigma: Tuple[float, float]
    pin_probability: float
    confidence_score: int
    max_profit_per_contract: float

    # Trap parameters
    trap_width: float
    trap_windows: List[dict]
    trap_trigger_pts: float
    trap_call_center: float
    trap_put_center: float
    trap_call_zone: Tuple[float, float]
    trap_put_zone: Tuple[float, float]

    # Option chains
    call_chain: str
    put_chain: str

    # Theta data
    theta_curve_data: List[dict]

    # P&L map
    pnl_map: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def generate_strategy(
    spx_open: float,
    vix: float,
    trade_date: date,
    entry_time: time = time(9, 35),
    expected_move_override: Optional[float] = None,
    risk_profile: RiskProfile = RiskProfile.MODERATE,
    regime_override: Optional[Regime] = None,
    calibration: Optional[dict] = None,
) -> StrategyRecommendation:
    """
    Main strategy generation function.

    Uses VIX, regime classification, day-of-week effects, and risk
    preferences to recommend batman position parameters.
    
    If calibration dict is provided (from real historical data analysis),
    the expected move is adjusted for VIX overstatement and uses
    data-driven DoW/month multipliers.
    """
    # Regime
    regime = regime_override or classify_regime(vix)
    rp = REGIME_PARAMS[regime]
    ra = RISK_ADJUSTMENTS[risk_profile]

    # Expected move — calibrated if real data available
    dow = trade_date.weekday()
    
    # RAW EM (VIX-implied) — always computed, used for WIDTH SIZING
    # This reflects the actual price range the market implies.
    # Width must cover this range regardless of VIX overstatement.
    raw_em = vix_to_daily_em(vix, spx_open)
    raw_dow_mult = DOW_VOL_MULTIPLIERS.get(dow, 1.0)
    raw_month_mult = MONTH_VOL_MULTIPLIERS.get(trade_date.month, 1.0)
    raw_em_adjusted = raw_em * raw_dow_mult * raw_month_mult
    
    # DISPLAY EM — calibrated for VIX overstatement if data available
    # This is what we show to the user and use for statistical analysis.
    if expected_move_override:
        em = expected_move_override
    elif calibration and "vix_overstatement" in calibration:
        overstatement = calibration["vix_overstatement"]
        adjusted_em = raw_em / overstatement
        
        dow_mults = calibration.get("dow_multipliers", {})
        dow_mult = dow_mults.get(trade_date.weekday(), 1.0)
        month_mults = calibration.get("month_multipliers", {})
        month_mult = month_mults.get(trade_date.month, 1.0)
        
        em = round(adjusted_em * dow_mult * month_mult, 1)
    else:
        em = round(raw_em_adjusted, 1)

    daily_sigma_pct = (vix / 100) / math.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Butterfly width = EM_WIDTH_FRACTION of RAW expected move ──
    # Risk profile adjusts the fraction:
    #   Conservative: 0.70 (wider = more coverage, higher debit)
    #   Moderate:     0.60 (backtested optimal)
    #   Aggressive:   0.50 (narrower = cheaper, higher R:R)
    EM_WIDTH_FRACTION = ra["em_width_frac"]
    bfly_width = round_to_strike(max(15, min(80, raw_em_adjusted * EM_WIDTH_FRACTION)))

    # ── Start offset = 0 (no dead zone) ──
    start_offset = 0

    # ── Strike calculation ──
    # Build butterflies outward from the open price:
    #   Inner wings at open (offset=0)
    #   Centers at open ± width
    #   Outer wings at open ± 2*width
    #
    # This matches the backtester's regime-tuned approach.
    # The EM is used for trap trigger zones, NOT for center placement.
    call_lower = round_to_strike(spx_open + start_offset)
    call_center = round_to_strike(call_lower + bfly_width)
    call_upper = call_center + bfly_width
    
    put_upper = round_to_strike(spx_open - start_offset)
    put_center = round_to_strike(put_upper - bfly_width)
    put_lower = put_center - bfly_width

    center_gap = call_lower - put_upper  # 0 with zero offset

    # ── Ranges ──
    r1 = (round_to_strike(spx_open - em), round_to_strike(spx_open + em))
    r2 = (round_to_strike(spx_open - 2 * em), round_to_strike(spx_open + 2 * em))

    # ── Pin probability ──
    # Probability SPX settles within the profit zone of either butterfly
    zone_low = put_center - bfly_width * rp.pin_zone_width
    zone_high = call_center + bfly_width * rp.pin_zone_width
    pin_prob = round(
        (norm_cdf((zone_high - spx_open) / em) - norm_cdf((zone_low - spx_open) / em)) * 100
    )

    # ── Confidence score ──
    confidence = rp.base_confidence + ra["conf_bonus"]
    if dow in (1, 3):  # Tue/Thu — calmer
        confidence += 3
    if dow in (0, 4):  # Mon/Fri — wilder
        confidence -= 2
    confidence = max(30, min(95, confidence))

    # ── Max profit ──
    max_profit = bfly_width  # points per contract

    # ── Trap parameters (optimized) ──
    # Trigger: price moves ≥ trigger_pct × EM from open
    # Trap width: 1.5× outer fly width (wider tent = more coverage)
    # Location: midpoint between open and fly center
    # Timing: 2:30 PM optimal (1.5h left, 70%+ theta burned)
    TRIGGER_PCT = {
        Regime.LOW_VOL: 0.15,
        Regime.NORMAL: 0.15,
        Regime.ELEVATED: 0.20,
        Regime.HIGH_VOL: 0.25,
    }
    trap_trigger_pct = TRIGGER_PCT.get(regime, 0.15) * ra["trigger_mult"]
    trap_trigger_pts = round(em * trap_trigger_pct, 1)
    trap_width = round_to_strike(max(10, bfly_width * ra["trap_width_mult"]))
    
    # Trap alert zones: price must reach ±trigger_pts from open
    trap_call_zone = (round(spx_open + trap_trigger_pts), round(spx_open + em * 0.6))
    trap_put_zone = (round(spx_open - em * 0.6), round(spx_open - trap_trigger_pts))
    
    # Midpoint trap centers (where trap fly would be placed)
    trap_call_center = round_to_strike((spx_open + call_center) / 2)
    trap_put_center = round_to_strike((spx_open + put_center) / 2)

    # ── Trap timing windows ──
    trap_windows = _generate_trap_windows(
        spx_open, em, bfly_width, rp, vix=vix,
        call_center=call_center, put_center=put_center,
        trap_width=trap_width, trap_trigger_pts=trap_trigger_pts,
        trap_call_center=trap_call_center, trap_put_center=trap_put_center,
    )

    # ── Option chain strings ──
    date_str = trade_date.strftime("%y%m%d")
    call_chain = (f".SPXW{date_str}C{int(call_lower)}-2*"
                  f".SPXW{date_str}C{int(call_center)}+"
                  f".SPXW{date_str}C{int(call_upper)}")
    put_chain = (f".SPXW{date_str}P{int(put_upper)}-2*"
                 f".SPXW{date_str}P{int(put_center)}+"
                 f".SPXW{date_str}P{int(put_lower)}")

    # ── Theta curve data ──
    tc = theta_curve(100)
    theta_data = [
        {"elapsed": round(t[0], 3), "cumulative_decay": round(t[1], 4),
         "rate": round(t[2], 3)}
        for t in tc
    ]

    # ── P&L map ──
    pnl_map = _generate_pnl_map(
        spx_open, em, call_lower, call_center, call_upper,
        put_lower, put_center, put_upper
    )

    return StrategyRecommendation(
        spx_open=spx_open, vix=vix, trade_date=trade_date.isoformat(),
        regime=regime.value, risk_profile=risk_profile.value,
        risk_profile_desc=ra["label"],
        butterfly_width=bfly_width, center_gap=center_gap,
        start_offset=start_offset,
        call_center=call_center, put_center=put_center,
        call_lower=call_lower, call_upper=call_upper,
        put_lower=put_lower, put_upper=put_upper,
        expected_move=em, raw_expected_move=round(raw_em_adjusted, 1),
        daily_sigma_pct=round(daily_sigma_pct * 100, 3),
        range_1sigma=r1, range_2sigma=r2,
        pin_probability=pin_prob, confidence_score=confidence,
        max_profit_per_contract=max_profit * 100,
        trap_width=trap_width, trap_windows=trap_windows,
        trap_trigger_pts=trap_trigger_pts,
        trap_call_center=trap_call_center, trap_put_center=trap_put_center,
        trap_call_zone=trap_call_zone, trap_put_zone=trap_put_zone,
        call_chain=call_chain, put_chain=put_chain,
        theta_curve_data=theta_data, pnl_map=pnl_map,
    )


def compute_trap_score(max_move_vs_em: float, range_expansion: float,
                       vix_change_5d: float) -> dict:
    """
    Compute real-time trap confidence score (0-100).
    
    Called at trap decision time (1:30-2:30 PM) with live data:
      max_move_vs_em:  max(|high-open|, |open-low|) / EM
      range_expansion: today's range / 5-day avg range
      vix_change_5d:   current VIX - 5-day avg VIX
    
    Returns dict with score, grade, recommendation, and trap sizing.
    
    Validated against 1,560 days:
      Score ≥ 60: PF=10.37, max DD reduced 28% vs always-trap
      Score < 40: avg P&L = -$9/day (these are the extended-move losers)
    """
    # Signal 1: Move magnitude (60% weight) — most predictive
    # Tiny/medium moves = butterflies cover the range = high WR
    # Extended moves = price blows past all wings = trap fails
    if max_move_vs_em < 0.3:
        move_score = 100
    elif max_move_vs_em < 0.7:
        move_score = 100 - (max_move_vs_em - 0.3) / 0.4 * 30
    elif max_move_vs_em < 1.0:
        move_score = 70 - (max_move_vs_em - 0.7) / 0.3 * 40
    elif max_move_vs_em < 1.3:
        move_score = 30 - (max_move_vs_em - 1.0) / 0.3 * 30
    else:
        move_score = 0
    
    # Signal 2: Range expansion (30% weight)
    # Normal/contracted range = mean reversion likely = trap holds
    # Expanded range = trending hard = trap risky
    if range_expansion < 0.6:
        range_score = 100
    elif range_expansion < 1.0:
        range_score = 100 - (range_expansion - 0.6) / 0.4 * 20
    elif range_expansion < 1.3:
        range_score = 80 - (range_expansion - 1.0) / 0.3 * 40
    elif range_expansion < 1.6:
        range_score = 40 - (range_expansion - 1.3) / 0.3 * 30
    else:
        range_score = 0
    
    # Signal 3: VIX trend (10% weight)
    # Falling VIX = calming market = trap friendly
    # Spiking VIX = new uncertainty = trap risky
    if vix_change_5d < -2:
        vix_score = 100
    elif vix_change_5d < 0:
        vix_score = 80
    elif vix_change_5d < 2:
        vix_score = 60
    else:
        vix_score = 30
    
    score = round(move_score * 0.60 + range_score * 0.30 + vix_score * 0.10)
    
    # Grade and recommendation
    if score >= 85:
        grade = "A"
        color = "green"
        label = "🟢 STRONG TRAP"
        action = "full"
        trap_width_mult = 1.5
        note = "High confidence. Full 1.5× width trap fly. Close losing side."
    elif score >= 60:
        grade = "B"
        color = "green" 
        label = "🟢 GOOD TRAP"
        action = "full"
        trap_width_mult = 1.5
        note = "Good confidence. Full 1.5× width trap fly. Close losing side."
    elif score >= 40:
        grade = "C"
        color = "amber"
        label = "🟡 CAUTIOUS"
        action = "half"
        trap_width_mult = 1.0
        note = "Moderate confidence. Conservative 1.0× width trap. Hold losing side."
    else:
        grade = "D"
        color = "red"
        label = "🔴 SKIP TRAP"
        action = "skip"
        trap_width_mult = 0
        note = "Extended move day. Hold batman position only. Do NOT add trap."
    
    return {
        "score": score,
        "grade": grade,
        "color": color,
        "label": label,
        "action": action,
        "trap_width_mult": trap_width_mult,
        "note": note,
        "signals": {
            "move_vs_em": round(max_move_vs_em, 2),
            "move_score": round(move_score),
            "range_expansion": round(range_expansion, 2),
            "range_score": round(range_score),
            "vix_change": round(vix_change_5d, 1),
            "vix_score": round(vix_score),
        }
    }


def _generate_trap_windows(spx: float, em: float, bfly_width: float,
                            rp: RegimeParams, vix: float = 16.0,
                            call_center: float = 0, put_center: float = 0,
                            trap_width: float = 0, trap_trigger_pts: float = 0,
                            trap_call_center: float = 0, trap_put_center: float = 0,
                            ) -> List[dict]:
    """
    Generate trap timing alerts with BS-calibrated debits.
    
    OPTIMIZED TRAP STRATEGY:
    ━━━━━━━━━━━━━━━━━━━━━━━━
    1. WATCH: Monitor SPX price from 11:30 AM onward
    2. TRIGGER: If SPX moves ≥ trigger_pts from open in either direction
    3. CONFIRM: Wait for price to hold direction (not a spike-reversal)
    4. TRAP: Buy 1.5× width butterfly centered at midpoint of open↔fly_center
    5. CLOSE: Sell the losing-side batman fly for ~15% credit
    6. HOLD: Combined position has wide profit tent covering most outcomes
    
    The user sees specific price alerts and exact strikes to enter.
    """
    from engine.backtester import _estimate_debit
    from engine.strategy import classify_regime
    
    regime = classify_regime(vix)
    tw = trap_width or round_to_strike(max(10, bfly_width * 1.5))
    
    # Compute trap debits at different times
    # Use the call-side trap center as representative
    tc = trap_call_center if trap_call_center else round_to_strike(spx + bfly_width * 0.5)
    
    call_alert = round(spx + trap_trigger_pts) if trap_trigger_pts else round(spx + em * 0.15)
    put_alert = round(spx - trap_trigger_pts) if trap_trigger_pts else round(spx - em * 0.15)
    
    windows = []
    
    # MONITORING phase
    windows.append({
        "time": "9:35 AM – 11:30 AM", 
        "label": "📊 Monitor & Hold",
        "action": "hold",
        "recommendation": "monitor",
        "risk": "—",
        "note": f"Hold batman position. Watch for SPX to break above {call_alert:.0f} or below {put_alert:.0f}.",
        "est_trap_debit": None,
        "theta_burned": 15,
        "condition": f"SPX between {put_alert:.0f} and {call_alert:.0f}",
    })
    
    # ALERT windows with specific debits
    for hours, label, time_str, theta, rec in [
        (4.5, "🔔 Early Trap Window",   "11:30 AM – 12:30 PM", 30, "cautious"),
        (3.0, "🔔 Primary Trap Window",  "12:30 – 1:30 PM",     45, "good"),
        (1.5, "⭐ Optimal Trap Window",  "1:30 – 2:30 PM",      70, "optimal"),
        (1.0, "⭐ Final Trap Window",    "2:30 – 3:15 PM",      85, "optimal"),
    ]:
        td = _estimate_debit(tw, vix, regime, float(tc), 
                             inner_wing=tc-tw, side="call", hours_left=hours)
        
        # Generate specific trap strikes for the alert
        call_trap_strikes = f"{int(trap_call_center-tw)}/{int(trap_call_center)}/{int(trap_call_center+tw)}"
        put_trap_strikes = f"{int(trap_put_center-tw)}/{int(trap_put_center)}/{int(trap_put_center+tw)}"
        
        windows.append({
            "time": time_str,
            "label": label,
            "action": "trap",
            "recommendation": rec,
            "risk": "medium" if hours > 3 else "low",
            "est_trap_debit": round(td, 2),
            "theta_burned": theta,
            "condition": f"SPX above {call_alert:.0f} OR below {put_alert:.0f}",
            "note": (f"Trap debit: ${td:.2f}/ct. "
                    f"If SPX↑: buy {call_trap_strikes} fly. "
                    f"If SPX↓: buy {put_trap_strikes} fly. "
                    f"Close losing side for ~15% credit."),
            "call_trap_strikes": call_trap_strikes,
            "put_trap_strikes": put_trap_strikes,
            "trap_width": tw,
        })
    
    # HEDGE window (for extreme move days)
    hedge_w = round_to_strike(max(10, bfly_width * 0.5))
    em_threshold = round(em)  # 1.0× EM
    hedge_call_strike = f"{int(spx + em_threshold)}/{int(spx + em_threshold + hedge_w)}"
    hedge_put_strike = f"{int(spx - em_threshold - hedge_w)}/{int(spx - em_threshold)}"
    
    windows.append({
        "time": "12:30 – 2:00 PM",
        "label": "🛡️ Directional Hedge (if extended)",
        "action": "hedge",
        "recommendation": "hedge",
        "risk": "low",
        "est_trap_debit": None,
        "theta_burned": 50,
        "condition": f"SPX above {round(spx + em_threshold)} OR below {round(spx - em_threshold)} (>1× EM) AND trap score RED",
        "note": (f"If trap score is 🔴 RED: Skip trap, buy directional spread instead. "
                f"SPX↑: buy {hedge_call_strike} call spread (~$0.50). "
                f"SPX↓: buy {hedge_put_strike} put spread (~$0.50). "
                f"Converts batman loss into directional winner."),
        "hedge_width": hedge_w,
    })
    
    # EXIT window
    windows.append({
        "time": "3:15 – 3:45 PM",
        "label": "🚪 Exit / Close",
        "action": "close",
        "recommendation": "close",
        "risk": "low",
        "est_trap_debit": None,
        "theta_burned": 95,
        "condition": "All positions",
        "note": "If trapped: hold to expiry (wide profit zone). If NOT trapped: close both sides to recover remaining value.",
    })
    
    return windows


def _generate_pnl_map(spx: float, em: float,
                       call_lower: float, call_center: float, call_upper: float,
                       put_lower: float, put_center: float, put_upper: float,
                       steps: int = 200) -> List[dict]:
    """Generate P&L across settlement prices."""
    # Range: from well below put_lower to well above call_upper
    # with enough margin to show the flat loss zones on both sides
    bfly_width = call_upper - call_center
    margin = max(bfly_width * 1.5, em * 0.5)
    range_low = put_lower - margin
    range_high = call_upper + margin
    results = []

    for i in range(steps + 1):
        price = range_low + (range_high - range_low) * i / steps

        call_pnl = (max(0, price - call_lower)
                     - 2 * max(0, price - call_center)
                     + max(0, price - call_upper))
        put_pnl = (max(0, put_upper - price)
                    - 2 * max(0, put_center - price)
                    + max(0, put_lower - price))

        results.append({
            "price": round(price, 1),
            "call_pnl": round(call_pnl, 2),
            "put_pnl": round(put_pnl, 2),
            "total_pnl": round(call_pnl + put_pnl, 2),
        })

    return results
