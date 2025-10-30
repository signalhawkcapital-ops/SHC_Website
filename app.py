# app.py
import os, logging
from datetime import datetime, date
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from flask import Flask, render_template, request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signal-hawk")

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("home.html", page_title="Signal Hawk Capital")

@app.route("/resources")
def resources():
    return render_template("resources_limina.html", page_title="Resources")

@app.route("/about")
def about():
    return render_template("about.html", page_title="About")

@app.route("/account")
def account():
    return render_template("account.html", page_title="Account")

@app.route("/signup")
def signup():
    return render_template("signup_limina.html", page_title="Subscribe")

TICKER_META = {
    "SPXL": {"label": "SPXL Buy & Hold", "expense_ratio": 0.0095},
    "SPY": {"label": "S&P 500 (SPY) Buy & Hold", "expense_ratio": 0.0009},
    "^GSPC": {"label": "S&P 500 Index (^GSPC) Buy & Hold", "expense_ratio": 0.0},
    "NVDA": {"label": "NVDA Buy & Hold", "expense_ratio": 0.0},
    "AMZN": {"label": "AMZN Buy & Hold", "expense_ratio": 0.0},
    "AAPL": {"label": "AAPL Buy & Hold", "expense_ratio": 0.0},
    "TSLA": {"label": "TSLA Buy & Hold", "expense_ratio": 0.0},
}

def _download_close(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.sort_index().ffill()

def _apply_expense_drag(series: pd.Series, ratio_annual: float) -> pd.Series:
    if ratio_annual <= 0: 
        return series
    daily_drag = (1 - ratio_annual) ** (1/252)
    ret = series.pct_change().fillna(0.0)
    adj_growth = (1 + ret) * daily_drag
    out = adj_growth.cumprod()
    return out * (series.iloc[0] / out.iloc[0])

def _equity_curve_from_price(price: pd.Series, initial: float, include_er: bool, expense_ratio: float,
                             include_taxes: bool, tax_rate: float = 0.2) -> pd.Series:
    px = price.copy().dropna()
    if include_er and expense_ratio > 0:
        px = _apply_expense_drag(px, expense_ratio)
    ret = px.pct_change().fillna(0.0)
    if include_taxes and tax_rate > 0:
        ret = np.where(ret > 0, ret * (1 - tax_rate), ret)
        ret = pd.Series(ret, index=px.index)
    equity = (1 + ret).cumprod() * (initial if initial else 10000.0)
    return equity

def _signal_hawk_equity(spx_price: pd.Series, spxl_price: pd.Series, initial: float,
                        include_er: bool, include_taxes: bool) -> pd.Series:
    spxl = spxl_price.copy().dropna()
    if include_er:
        spxl = _apply_expense_drag(spxl, TICKER_META["SPXL"]["expense_ratio"])
    spx = spx_price.reindex_like(spxl).ffill()
    sma = spx.rolling(80).mean()
    signal = (spx > sma).astype(int).shift(1).fillna(0)
    ret_spxl = spxl.pct_change().fillna(0.0)
    strat_ret = ret_spxl * signal
    if include_taxes:
        strat_ret = np.where(strat_ret > 0, strat_ret * (1 - 0.2), strat_ret)
        strat_ret = pd.Series(strat_ret, index=spxl.index)
    equity = (1 + strat_ret).cumprod() * (initial if initial else 10000.0)
    equity.name = "Signal Hawk Strategy"
    return equity

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    today = date.today()
    start = request.values.get("start", "2010-01-01")
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
    except ValueError:
        start_date = date(2010, 1, 1)

    initial = float(request.values.get("initial", 10000))
    include_er = request.values.get("include_er", "on") == "on"
    include_taxes = request.values.get("include_taxes", "off") == "on"

    defaults = {
        "series_SH": "on",
        "series_HG_SIM": "on",
        "series_SPXL": "on",
        "series_SP500": "on",
        "series_NVDA": "off",
        "series_AMZN": "off",
        "series_AAPL": "off",
        "series_TSLA": "off",
    }
    series_flags = {k: request.values.get(k, defaults[k]) == "on" for k in defaults}

    need = ["SPXL", "SPY", "^GSPC"]
    for sym, flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
        if series_flags[flag]:
            need.append(sym)
    df_close = _download_close(sorted(set(need)), start_date, today)

    curves = {}
    spx = df_close.get("^GSPC") if "^GSPC" in df_close.columns else df_close.get("SPY")
    if isinstance(spx, pd.DataFrame): spx = spx.iloc[:,0]
    spxl = df_close.get("SPXL", None)
    if isinstance(spxl, pd.DataFrame): spxl = spxl.iloc[:,0]
    spy = df_close.get("SPY", None)
    if isinstance(spy, pd.DataFrame): spy = spy.iloc[:,0]

    if series_flags["series_SH"] and spx is not None and spxl is not None:
        curves["Signal Hawk Strategy"] = _signal_hawk_equity(spx, spxl, initial, include_er, include_taxes)

    if series_flags["series_HG_SIM"] and spx is not None and spy is not None:
        sma = spx.rolling(80).mean()
        signal = (spx > sma).astype(int).shift(1).fillna(0)
        ret_spy = spy.pct_change().fillna(0.0) * 2.5
        if include_er:
            ret_spy = pd.Series(np.where(ret_spy>0, ret_spy*(1 - 0.0095/0.4), ret_spy), index=spy.index)
        if include_taxes:
            ret_spy = pd.Series(np.where(ret_spy>0, ret_spy*(1-0.2), ret_spy), index=spy.index)
        curves["HG_SIM Strategy"] = (1 + ret_spy*signal).cumprod() * initial

    if series_flags["series_SPXL"] and spxl is not None:
        curves["SPXL Buy & Hold"] = _equity_curve_from_price(spxl, initial, include_er, 0.0095, include_taxes)

    spx_source = "^GSPC" if "^GSPC" in df_close.columns else "SPY"
    if series_flags["series_SP500"] and spx_source in df_close.columns:
        series = df_close[spx_source]
        if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
        curves[("S&P 500 (SPY) Buy & Hold" if spx_source=="SPY" else "S&P 500 Index (^GSPC) Buy & Hold")] = _equity_curve_from_price(series, initial, include_er, 0.0009 if spx_source=="SPY" else 0.0, include_taxes)

    for sym, flag in [("NVDA","series_NVDA"),("AMZN","series_AMZN"),("AAPL","series_AAPL"),("TSLA","series_TSLA")]:
        if series_flags[flag] and sym in df_close.columns:
            series = df_close[sym]
            if isinstance(series, pd.DataFrame): series = series.iloc[:,0]
            curves[f"{sym} Buy & Hold"] = _equity_curve_from_price(series, initial, include_er, 0.0, include_taxes)

    color_map = {
        "Signal Hawk Strategy": "#3b5bdb",
        "HG_SIM Strategy": "#6f42c1",
        "SPXL Buy & Hold": "#fd7e14",
        "S&P 500 (SPY) Buy & Hold": "#2fbf71",
        "S&P 500 Index (^GSPC) Buy & Hold": "#2fbf71",
        "NVDA Buy & Hold": "#00bcd4",
        "AMZN Buy & Hold": "#9c27b0",
        "AAPL Buy & Hold": "#607d8b",
        "TSLA Buy & Hold": "#e91e63",
    }

    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name,
                                 line=dict(width=2, color=color_map.get(name))))
    fig.update_layout(
        title="Investment Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=600,
        template="plotly_dark",
        margin=dict(l=30,r=20,t=30,b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    price_graph = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    table_rows = []
    for name, series in curves.items():
        years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1 if years > 0 else float('nan')
        mdd = (series / series.cummax() - 1).min()
        table_rows.append({"name": name, "cagr": f"{cagr*100:.2f}%", "mdd": f"{mdd*100:.2f}%", "final": f"${series.iloc[-1]:,.0f}"})

    msg = f"If you invested ${initial:,.2f} on {start_date.strftime('%m/%d/%Y')} and used the Signal Hawk buy & sell alerts your account value would be "
    if "Signal Hawk Strategy" in curves:
        msg += f"${curves['Signal Hawk Strategy'].iloc[-1]:,.2f} by {date.today().strftime('%m/%d/%Y')}.*"
    else:
        msg += "— (insufficient data).*"

    return render_template("growth_analyzer.html",
                           page_title="Investment Growth Analyzer",
                           price_graph=price_graph,
                           message=msg,
                           start=start_date.strftime("%Y-%m-%d"),
                           initial=initial,
                           include_er=include_er,
                           include_taxes=include_taxes,
                           series_flags=series_flags,
                           table_rows=table_rows)

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug)
