"""
Directional Signals Engine
============================
4 evidence-based intraday strategies for SPX 0DTE directional option buying.

Each strategy produces: direction (CALL/PUT), entry time, strike selection,
stop loss, profit target, and a confidence score. All are backtested against
real daily OHLC data.

STRATEGIES:
1. Opening Range Breakout (ORB) — 30-min range breakout with trend filter
2. Mean Reversion Fade — Oversold/overbought at prior day's key levels
3. VWAP Momentum — Price vs VWAP with VIX regime filter
4. Gap & Follow — Overnight gap direction continuation/fade

IMPORTANT DISCLAIMERS:
- These are statistical models, NOT guaranteed profitable strategies
- Past backtest performance does NOT guarantee future results
- Real trading involves slippage, spreads, and execution risk not modeled
- Position sizing and risk management are the user's responsibility
"""

import math
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from datetime import date, timedelta
from enum import Enum

from engine.strategy import vix_to_daily_em, classify_regime, Regime


class Direction(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    NONE = "NONE"


@dataclass
class Signal:
    strategy: str
    direction: Direction
    confidence: float          # 0-100
    entry_time: str            # e.g. "10:05 AM"
    strike_offset: float       # pts from open (+ = OTM call, - = OTM put)
    stop_pct: float            # % of premium to risk
    target_pct: float          # % of premium to target
    rationale: str
    conditions_met: List[str]
    conditions_failed: List[str] = field(default_factory=list)


@dataclass
class DirectionalSummary:
    date: str
    spx_open: float
    vix: float
    regime: str
    signals: List[dict]
    consensus_direction: str
    consensus_confidence: float
    recommended_strike: float
    recommended_entry: str
    risk_reward: str
    backtest_stats: dict
    disclaimer: str = ("Statistical model for educational purposes only. "
                       "Not financial advice. Past performance ≠ future results.")

    def to_dict(self):
        return asdict(self)


# ────────────────────────────────────────────────────────────
# Strategy 1: Opening Range Breakout (ORB)
# ────────────────────────────────────────────────────────────

def _orb_signal(bar, prev_bar, prev_bars, vix) -> Signal:
    """
    30-minute Opening Range Breakout.

    Logic: Define the range from open to the high/low of the first 30 min
    (approximated from daily data as: range = open ± (day_range * 0.35)).
    If close breaks above the range → CALL, below → PUT.

    Filters:
    - Prior day trend (close vs open direction)
    - VIX regime (avoid high vol for trend-following)
    - Gap size (small gaps favor ORB)
    """
    em = vix_to_daily_em(vix, bar.spx_open)
    orb_range = em * 0.4  # Approximate 30-min range as 40% of daily EM

    orb_high = bar.spx_open + orb_range
    orb_low = bar.spx_open - orb_range

    conditions_met = []
    conditions_failed = []
    direction = Direction.NONE
    confidence = 0

    # Direction from close vs ORB
    if bar.spx_close > orb_high:
        direction = Direction.CALL
        conditions_met.append(f"Close {bar.spx_close:.0f} > ORB high {orb_high:.0f}")
    elif bar.spx_close < orb_low:
        direction = Direction.PUT
        conditions_met.append(f"Close {bar.spx_close:.0f} < ORB low {orb_low:.0f}")
    else:
        conditions_failed.append("Price stayed within ORB range — no breakout")
        return Signal("ORB", Direction.NONE, 0, "10:05 AM", 0, 50, 100,
                      "No breakout detected", [], conditions_failed)

    confidence = 45

    # Filter 1: Prior day trend alignment
    if prev_bar:
        prev_trend = "up" if prev_bar.spx_close > prev_bar.spx_open else "down"
        if (direction == Direction.CALL and prev_trend == "up") or \
           (direction == Direction.PUT and prev_trend == "down"):
            confidence += 15
            conditions_met.append(f"Prior day trend aligns ({prev_trend})")
        else:
            confidence -= 5
            conditions_failed.append(f"Prior day trend opposes ({prev_trend})")

    # Filter 2: VIX regime — ORB works best in normal/low vol
    regime = classify_regime(vix)
    if regime in (Regime.LOW_VOL, Regime.NORMAL):
        confidence += 10
        conditions_met.append(f"VIX {vix:.1f} in favorable regime")
    elif regime == Regime.ELEVATED:
        confidence -= 5
        conditions_failed.append(f"VIX {vix:.1f} elevated — higher false breakout risk")
    else:
        confidence -= 15
        conditions_failed.append(f"VIX {vix:.1f} high — ORB unreliable")

    # Filter 3: Gap — small gaps favor ORB
    if prev_bar:
        gap = abs(bar.spx_open - prev_bar.spx_close)
        gap_pct = gap / bar.spx_open * 100
        if gap_pct < 0.3:
            confidence += 8
            conditions_met.append(f"Small gap ({gap_pct:.2f}%) favors ORB")
        elif gap_pct > 0.8:
            confidence -= 10
            conditions_failed.append(f"Large gap ({gap_pct:.2f}%) — ORB less reliable")

    # Filter 4: Breakout strength
    if direction == Direction.CALL:
        strength = (bar.spx_close - orb_high) / em * 100
    else:
        strength = (orb_low - bar.spx_close) / em * 100
    if strength > 30:
        confidence += 8
        conditions_met.append(f"Strong breakout ({strength:.0f}% of EM)")
    elif strength < 10:
        confidence -= 5

    confidence = max(0, min(95, confidence))
    strike_offset = round(em * 0.15, 0) if direction == Direction.CALL else -round(em * 0.15, 0)

    return Signal(
        strategy="ORB",
        direction=direction,
        confidence=round(confidence),
        entry_time="10:05 AM",
        strike_offset=strike_offset,
        stop_pct=40,
        target_pct=100,
        rationale=f"30-min ORB breakout {'above' if direction == Direction.CALL else 'below'} "
                  f"{orb_high:.0f}/{orb_low:.0f}",
        conditions_met=conditions_met,
        conditions_failed=conditions_failed,
    )


# ────────────────────────────────────────────────────────────
# Strategy 2: Mean Reversion Fade
# ────────────────────────────────────────────────────────────

def _mean_reversion_signal(bar, prev_bar, prev_bars, vix) -> Signal:
    """
    Fade extreme opening moves back toward the mean.

    Logic: If the market gaps or moves sharply in one direction from
    prior close and hits an oversold/overbought level, fade it.

    Best in: elevated/high VIX (mean reversion stronger), range-bound days.
    """
    if not prev_bar:
        return Signal("MR Fade", Direction.NONE, 0, "10:30 AM", 0, 50, 80,
                      "Insufficient data", [], ["No prior bar"])

    em = vix_to_daily_em(vix, bar.spx_open)
    gap = bar.spx_open - prev_bar.spx_close
    gap_pct = gap / bar.spx_open * 100

    conditions_met = []
    conditions_failed = []
    direction = Direction.NONE
    confidence = 0

    # Need a significant gap or opening move
    if abs(gap_pct) < 0.15:
        conditions_failed.append(f"Gap too small ({gap_pct:+.2f}%) — no fade setup")
        return Signal("MR Fade", Direction.NONE, 0, "10:30 AM", 0, 50, 80,
                      "No extreme move to fade", [], conditions_failed)

    # RSI proxy: use N-day momentum to judge if extended
    if len(prev_bars) >= 3:
        recent_returns = [(prev_bars[i].spx_close - prev_bars[i].spx_open) / prev_bars[i].spx_open
                          for i in range(min(5, len(prev_bars)))]
        cum_move = sum(recent_returns) * 100

        if gap_pct > 0.2 and cum_move > 0.5:
            direction = Direction.PUT
            confidence = 50
            conditions_met.append(f"Overbought: gap {gap_pct:+.2f}%, 3-day run {cum_move:+.2f}%")
        elif gap_pct < -0.2 and cum_move < -0.5:
            direction = Direction.CALL
            confidence = 50
            conditions_met.append(f"Oversold: gap {gap_pct:+.2f}%, 3-day run {cum_move:+.2f}%")
        elif abs(gap_pct) > 0.5:
            # Large gap alone is fadeable
            direction = Direction.CALL if gap_pct < 0 else Direction.PUT
            confidence = 40
            conditions_met.append(f"Large gap ({gap_pct:+.2f}%) — fade likely")
        else:
            conditions_failed.append("Not sufficiently extended for mean reversion")
            return Signal("MR Fade", Direction.NONE, 0, "10:30 AM", 0, 50, 80,
                          "Conditions not met", [], conditions_failed)

    # Filter: VIX regime — mean reversion STRONGER in high vol
    regime = classify_regime(vix)
    if regime in (Regime.ELEVATED, Regime.HIGH_VOL):
        confidence += 12
        conditions_met.append(f"High VIX ({vix:.1f}) favors mean reversion")
    elif regime == Regime.NORMAL:
        confidence += 5
    else:
        confidence -= 8
        conditions_failed.append(f"Low VIX ({vix:.1f}) — trending market, fade riskier")

    # Filter: Did prior day close near its high/low? (exhaustion signal)
    if prev_bar:
        day_range = prev_bar.spx_high - prev_bar.spx_low
        if day_range > 0:
            close_position = (prev_bar.spx_close - prev_bar.spx_low) / day_range
            if direction == Direction.PUT and close_position > 0.85:
                confidence += 10
                conditions_met.append("Prior day closed near high — exhaustion")
            elif direction == Direction.CALL and close_position < 0.15:
                confidence += 10
                conditions_met.append("Prior day closed near low — exhaustion")

    confidence = max(0, min(95, confidence))
    strike_offset = round(em * 0.10, 0) if direction == Direction.CALL else -round(em * 0.10, 0)

    return Signal(
        strategy="MR Fade",
        direction=direction,
        confidence=round(confidence),
        entry_time="10:30 AM",
        strike_offset=strike_offset,
        stop_pct=50,
        target_pct=80,
        rationale=f"Fade {'gap up' if gap > 0 else 'gap down'} — "
                  f"mean reversion from extended {abs(gap_pct):.2f}% move",
        conditions_met=conditions_met,
        conditions_failed=conditions_failed,
    )


# ────────────────────────────────────────────────────────────
# Strategy 3: VWAP Momentum
# ────────────────────────────────────────────────────────────

def _vwap_momentum_signal(bar, prev_bar, prev_bars, vix) -> Signal:
    """
    VWAP-based momentum: trade in direction of price vs VWAP proxy.

    We approximate intraday VWAP position using: if close > midpoint and
    close > open → above VWAP. Combined with multi-day momentum.

    Best in: normal VIX, trending days.
    """
    em = vix_to_daily_em(vix, bar.spx_open)
    midpoint = (bar.spx_high + bar.spx_low) / 2

    conditions_met = []
    conditions_failed = []
    direction = Direction.NONE
    confidence = 0

    # VWAP proxy: close vs midpoint (institutional fair value)
    if bar.spx_close > midpoint and bar.spx_close > bar.spx_open:
        direction = Direction.CALL
        confidence = 40
        conditions_met.append(f"Close > midpoint & above open — bullish VWAP position")
    elif bar.spx_close < midpoint and bar.spx_close < bar.spx_open:
        direction = Direction.PUT
        confidence = 40
        conditions_met.append(f"Close < midpoint & below open — bearish VWAP position")
    else:
        conditions_failed.append("Mixed VWAP signals — no clear momentum")
        return Signal("VWAP Mom", Direction.NONE, 0, "11:00 AM", 0, 45, 90,
                      "No clear VWAP momentum", [], conditions_failed)

    # Multi-day trend alignment (3-day EMA proxy)
    if len(prev_bars) >= 3:
        ema3 = np.mean([b.spx_close for b in prev_bars[:3]])
        if direction == Direction.CALL and bar.spx_open > ema3:
            confidence += 12
            conditions_met.append(f"Above 3-day EMA ({ema3:.0f}) — trend aligned")
        elif direction == Direction.PUT and bar.spx_open < ema3:
            confidence += 12
            conditions_met.append(f"Below 3-day EMA ({ema3:.0f}) — trend aligned")
        else:
            confidence -= 5
            conditions_failed.append("Counter-trend — lower conviction")

    # Range position: strong days close near high/low
    day_range = bar.spx_high - bar.spx_low
    if day_range > 0:
        close_pct = (bar.spx_close - bar.spx_low) / day_range
        if direction == Direction.CALL and close_pct > 0.7:
            confidence += 10
            conditions_met.append(f"Closed in upper 30% of range — strong momentum")
        elif direction == Direction.PUT and close_pct < 0.3:
            confidence += 10
            conditions_met.append(f"Closed in lower 30% of range — strong momentum")

    # VIX filter
    regime = classify_regime(vix)
    if regime == Regime.NORMAL:
        confidence += 5
        conditions_met.append("Normal VIX — trend following favorable")
    elif regime == Regime.HIGH_VOL:
        confidence -= 10
        conditions_failed.append("High VIX — momentum whipsaws more likely")

    confidence = max(0, min(95, confidence))
    strike_offset = round(em * 0.20, 0) if direction == Direction.CALL else -round(em * 0.20, 0)

    return Signal(
        strategy="VWAP Mom",
        direction=direction,
        confidence=round(confidence),
        entry_time="11:00 AM",
        strike_offset=strike_offset,
        stop_pct=45,
        target_pct=90,
        rationale=f"{'Bullish' if direction == Direction.CALL else 'Bearish'} momentum — "
                  f"price {'above' if direction == Direction.CALL else 'below'} VWAP proxy",
        conditions_met=conditions_met,
        conditions_failed=conditions_failed,
    )


# ────────────────────────────────────────────────────────────
# Strategy 4: Gap & Follow / Fade
# ────────────────────────────────────────────────────────────

def _gap_signal(bar, prev_bar, prev_bars, vix) -> Signal:
    """
    Gap analysis: continuation vs fade based on gap size and context.

    Small gaps (< 0.3%) → tend to fill (fade)
    Large gaps (> 0.5%) with trend → tend to continue
    Large gaps against trend → tend to fill

    Based on: Gaps fill ~70% of the time within the day for SPX,
    but large trend-aligned gaps continue ~55% of the time.
    """
    if not prev_bar:
        return Signal("Gap Play", Direction.NONE, 0, "9:45 AM", 0, 50, 80,
                      "No prior data", [], ["Missing prior bar"])

    gap = bar.spx_open - prev_bar.spx_close
    gap_pct = gap / bar.spx_open * 100
    em = vix_to_daily_em(vix, bar.spx_open)

    conditions_met = []
    conditions_failed = []
    direction = Direction.NONE
    confidence = 0

    if abs(gap_pct) < 0.05:
        conditions_failed.append("Negligible gap — no setup")
        return Signal("Gap Play", Direction.NONE, 0, "9:45 AM", 0, 50, 80,
                      "No meaningful gap", [], conditions_failed)

    # Check multi-day trend
    trend_up = True
    if len(prev_bars) >= 3:
        trend_up = prev_bars[0].spx_close > prev_bars[2].spx_close

    # Gap classification
    if abs(gap_pct) < 0.3:
        # Small gap → fade (gaps fill ~70% of time)
        direction = Direction.PUT if gap > 0 else Direction.CALL
        confidence = 50
        conditions_met.append(f"Small gap ({gap_pct:+.2f}%) — gap fill expected (~70% hist.)")
    elif abs(gap_pct) >= 0.5:
        # Large gap — check trend alignment
        gap_up = gap > 0
        if gap_up == trend_up:
            # Gap with trend → continuation
            direction = Direction.CALL if gap_up else Direction.PUT
            confidence = 45
            conditions_met.append(f"Large gap ({gap_pct:+.2f}%) WITH trend — continuation")
        else:
            # Gap against trend → fade
            direction = Direction.PUT if gap_up else Direction.CALL
            confidence = 52
            conditions_met.append(f"Large gap ({gap_pct:+.2f}%) AGAINST trend — fade likely")
    else:
        # Medium gap — lean fade
        direction = Direction.PUT if gap > 0 else Direction.CALL
        confidence = 42
        conditions_met.append(f"Medium gap ({gap_pct:+.2f}%) — slight fade bias")

    # Did price actually fill the gap?
    if gap > 0 and bar.spx_low <= prev_bar.spx_close:
        if direction == Direction.PUT:
            confidence += 10
            conditions_met.append("Gap DID fill — fade confirmed")
    elif gap < 0 and bar.spx_high >= prev_bar.spx_close:
        if direction == Direction.CALL:
            confidence += 10
            conditions_met.append("Gap DID fill — fade confirmed")

    # VIX context
    regime = classify_regime(vix)
    if regime in (Regime.ELEVATED, Regime.HIGH_VOL) and abs(gap_pct) < 0.5:
        confidence += 5
        conditions_met.append("Higher VIX favors gap fill")

    confidence = max(0, min(95, confidence))
    strike_offset = round(em * 0.10, 0) if direction == Direction.CALL else -round(em * 0.10, 0)

    return Signal(
        strategy="Gap Play",
        direction=direction,
        confidence=round(confidence),
        entry_time="9:45 AM",
        strike_offset=strike_offset,
        stop_pct=50,
        target_pct=80,
        rationale=f"Gap {gap_pct:+.2f}% — {'fade' if (gap > 0) == (direction == Direction.PUT) else 'follow'}",
        conditions_met=conditions_met,
        conditions_failed=conditions_failed,
    )


# ────────────────────────────────────────────────────────────
# Signal Aggregator
# ────────────────────────────────────────────────────────────

def generate_directional_signals(bar, prev_bars, vix=None) -> DirectionalSummary:
    """Generate all directional signals for a given day."""
    if vix is None:
        vix = bar.vix_open

    prev_bar = prev_bars[0] if prev_bars else None

    signals = [
        _orb_signal(bar, prev_bar, prev_bars, vix),
        _mean_reversion_signal(bar, prev_bar, prev_bars, vix),
        _vwap_momentum_signal(bar, prev_bar, prev_bars, vix),
        _gap_signal(bar, prev_bar, prev_bars, vix),
    ]

    # Consensus: weighted vote by confidence
    call_score = sum(s.confidence for s in signals if s.direction == Direction.CALL)
    put_score = sum(s.confidence for s in signals if s.direction == Direction.PUT)
    active_signals = [s for s in signals if s.direction != Direction.NONE]

    if call_score > put_score and call_score > 0:
        consensus = "CALL"
        consensus_conf = round(call_score / max(call_score + put_score, 1) * 100)
    elif put_score > call_score and put_score > 0:
        consensus = "PUT"
        consensus_conf = round(put_score / max(call_score + put_score, 1) * 100)
    else:
        consensus = "NEUTRAL"
        consensus_conf = 0

    # Best signal for strike recommendation
    best = max(active_signals, key=lambda s: s.confidence) if active_signals else None
    em = vix_to_daily_em(vix, bar.spx_open)

    if best and best.direction != Direction.NONE:
        if best.direction == Direction.CALL:
            rec_strike = round(bar.spx_open + abs(best.strike_offset))
        else:
            rec_strike = round(bar.spx_open - abs(best.strike_offset))
        rec_strike = round(rec_strike / 5) * 5
        rec_entry = best.entry_time
        rr = f"Risk {best.stop_pct}% / Target {best.target_pct}%"
    else:
        rec_strike = round(bar.spx_open / 5) * 5
        rec_entry = "N/A"
        rr = "No trade"

    return DirectionalSummary(
        date=bar.trade_date.isoformat(),
        spx_open=bar.spx_open,
        vix=vix,
        regime=classify_regime(vix).value,
        signals=[asdict(s) for s in signals],
        consensus_direction=consensus,
        consensus_confidence=consensus_conf,
        recommended_strike=rec_strike,
        recommended_entry=rec_entry,
        risk_reward=rr,
        backtest_stats={},
    )


# ────────────────────────────────────────────────────────────
# Backtester for Directional Strategies
# ────────────────────────────────────────────────────────────

@dataclass
class DirectionalBacktestResult:
    strategy: str
    total_signals: int
    call_signals: int
    put_signals: int
    correct: int
    wrong: int
    no_signal: int
    win_rate: float
    avg_move_when_correct: float
    avg_move_when_wrong: float
    profit_factor: float
    avg_rr_pts: float
    best_regime: str
    worst_regime: str
    best_dow: str
    worst_dow: str
    notes: List[str]

    def to_dict(self):
        return asdict(self)


def backtest_directional(bars, strategy_fn, strategy_name) -> DirectionalBacktestResult:
    """Backtest a single directional strategy against historical data."""
    results = []

    for i in range(5, len(bars)):
        bar = bars[i]
        prev_bars = list(reversed(bars[max(0, i - 5):i]))
        vix = bar.vix_open

        signal = strategy_fn(bar, prev_bars[0] if prev_bars else None, prev_bars, vix)

        if signal.direction == Direction.NONE:
            results.append({"direction": "NONE", "correct": None, "move": 0,
                            "regime": classify_regime(vix).value,
                            "dow": bar.trade_date.weekday()})
            continue

        actual_move = bar.spx_close - bar.spx_open
        correct = (signal.direction == Direction.CALL and actual_move > 0) or \
                  (signal.direction == Direction.PUT and actual_move < 0)

        # P&L proxy: if correct, you capture the move (capped at target)
        # If wrong, you lose the stop amount
        em = vix_to_daily_em(vix, bar.spx_open)
        target_pts = em * signal.target_pct / 100
        stop_pts = em * signal.stop_pct / 100

        if correct:
            rr_pts = min(abs(actual_move), target_pts)
        else:
            rr_pts = -min(abs(actual_move), stop_pts)

        results.append({
            "direction": signal.direction.value,
            "correct": correct,
            "move": actual_move,
            "rr_pts": rr_pts,
            "regime": classify_regime(vix).value,
            "dow": bar.trade_date.weekday(),
            "confidence": signal.confidence,
        })

    # Aggregate
    trades = [r for r in results if r["direction"] != "NONE"]
    no_signal = len(results) - len(trades)
    if not trades:
        return DirectionalBacktestResult(
            strategy=strategy_name, total_signals=0, call_signals=0, put_signals=0,
            correct=0, wrong=0, no_signal=no_signal, win_rate=0,
            avg_move_when_correct=0, avg_move_when_wrong=0, profit_factor=0,
            avg_rr_pts=0, best_regime="N/A", worst_regime="N/A",
            best_dow="N/A", worst_dow="N/A", notes=["No signals generated"])

    correct_trades = [t for t in trades if t["correct"]]
    wrong_trades = [t for t in trades if not t["correct"]]

    win_rate = len(correct_trades) / len(trades) * 100 if trades else 0

    avg_correct = float(np.mean([abs(t["move"]) for t in correct_trades])) if correct_trades else 0
    avg_wrong = float(np.mean([abs(t["move"]) for t in wrong_trades])) if wrong_trades else 0

    gross_profit = sum(t.get("rr_pts", 0) for t in correct_trades)
    gross_loss = abs(sum(t.get("rr_pts", 0) for t in wrong_trades))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99

    avg_rr = float(np.mean([t.get("rr_pts", 0) for t in trades]))

    # Regime breakdown
    regime_wr = {}
    for regime in ["low_vol", "normal", "elevated", "high_vol"]:
        rt = [t for t in trades if t["regime"] == regime]
        if len(rt) >= 5:
            regime_wr[regime] = len([t for t in rt if t["correct"]]) / len(rt) * 100
    best_regime = max(regime_wr, key=regime_wr.get) if regime_wr else "N/A"
    worst_regime = min(regime_wr, key=regime_wr.get) if regime_wr else "N/A"

    # DoW breakdown
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_wr = {}
    for d in range(5):
        dt = [t for t in trades if t["dow"] == d]
        if len(dt) >= 5:
            dow_wr[dow_names[d]] = len([t for t in dt if t["correct"]]) / len(dt) * 100
    best_dow = max(dow_wr, key=dow_wr.get) if dow_wr else "N/A"
    worst_dow = min(dow_wr, key=dow_wr.get) if dow_wr else "N/A"

    notes = []
    if win_rate > 55:
        notes.append(f"✅ {win_rate:.1f}% win rate — positive edge detected")
    elif win_rate > 48:
        notes.append(f"📊 {win_rate:.1f}% win rate — marginal edge, needs good R:R")
    else:
        notes.append(f"⚠ {win_rate:.1f}% win rate — below breakeven")

    if pf > 1.2:
        notes.append(f"✅ Profit factor {pf:.2f} — strategy is net profitable")
    elif pf > 1.0:
        notes.append(f"📊 Profit factor {pf:.2f} — barely profitable after costs")
    else:
        notes.append(f"⚠ Profit factor {pf:.2f} — net loss")

    if best_regime != "N/A":
        notes.append(f"Best regime: {best_regime} ({regime_wr.get(best_regime, 0):.0f}% WR)")
    if best_dow != "N/A":
        notes.append(f"Best day: {best_dow} ({dow_wr.get(best_dow, 0):.0f}% WR)")

    return DirectionalBacktestResult(
        strategy=strategy_name,
        total_signals=len(trades),
        call_signals=len([t for t in trades if t["direction"] == "CALL"]),
        put_signals=len([t for t in trades if t["direction"] == "PUT"]),
        correct=len(correct_trades),
        wrong=len(wrong_trades),
        no_signal=no_signal,
        win_rate=round(win_rate, 1),
        avg_move_when_correct=round(avg_correct, 1),
        avg_move_when_wrong=round(avg_wrong, 1),
        profit_factor=round(pf, 2),
        avg_rr_pts=round(avg_rr, 2),
        best_regime=best_regime,
        worst_regime=worst_regime,
        best_dow=best_dow,
        worst_dow=worst_dow,
        notes=notes,
    )


def backtest_all_directional(bars) -> List[DirectionalBacktestResult]:
    """Run all 4 directional strategy backtests."""
    strategies = [
        (_orb_signal, "ORB"),
        (_mean_reversion_signal, "MR Fade"),
        (_vwap_momentum_signal, "VWAP Mom"),
        (_gap_signal, "Gap Play"),
    ]
    return [backtest_directional(bars, fn, name) for fn, name in strategies]
