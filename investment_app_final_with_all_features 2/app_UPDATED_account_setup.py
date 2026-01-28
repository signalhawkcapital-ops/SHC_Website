import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for, send_file
from markupsafe import Markup
import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime, timedelta
import time


from zoneinfo import ZoneInfo

def get_previous_trading_day_end():
    """Return a datetime at midnight for the previous trading day (US/Eastern) based on ^GSPC availability."""
    # We deliberately use yfinance to check actual available daily bars.
    import yfinance as yf
    from datetime import datetime
    import pandas as pd

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    # Fetch a buffer of days to safely find the most recent session strictly before today
    hist = yf.download("^GSPC", period="15d", interval="1d", progress=False, auto_adjust=False)
    if hist is None or hist.empty:
        # Fallback: assume previous weekday if data unavailable
        prev = today_et
        # step back at least one day
        from datetime import timedelta
        prev -= timedelta(days=1)
        # roll back over weekends
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        return datetime.combine(prev, datetime.min.time())

    # Convert index to date (ensure tz-naive daily index)
    dates = [idx.date() for idx in hist.index]
    # Choose the max date strictly before today (close of previous session)
    prior_dates = [d for d in dates if d < today_et]
    if not prior_dates:
        # Edge case: if run very early ET and yesterday isn't available yet, step back 1 business day
        from datetime import timedelta
        prev = today_et - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        return datetime.combine(prev, datetime.min.time())
    prev = max(prior_dates)
    return datetime.combine(prev, datetime.min.time())

app = Flask(__name__)
pio.templates.default = "plotly_dark"


def download_data(tickers, start, end, retries=8, base_delay=2.0):
    """Robust close-price downloader with per‑ticker retries, fallbacks, and on-disk cache.
    Returns a DataFrame of 'Close' columns by ticker, indexed by date.
    Creates ./data_cache/<TICKER>.csv per ticker for resiliency.
    """
    import os, time, math, random
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from datetime import datetime

    # Normalize tickers -> list[str]
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    cache_dir = Path("./data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _from_cache(t):
        f = cache_dir / f"{t.replace('^','_')}.csv"
        if f.exists():
            try:
                df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
                # Only keep Close if present
                if "Close" in df.columns:
                    return df[["Close"]].rename(columns={"Close": t})
                # Some older caches may store columns without rename — handle both cases
                if t in df.columns:
                    return df[[t]].rename(columns={t: t})
            except Exception as _:
                pass
        return None

    def _to_cache(t, df_close):
        try:
            out = df_close.rename(columns={t: "Close"}).copy()
            out.index.name = "Date"
            out.reset_index().to_csv(cache_dir / f"{t.replace('^','_')}.csv", index=False)
        except Exception as _:
            pass

    def _attempt_download(t, start, end):
        # Try several strategies per attempt
        # 1) yf.download for the specific date window
        # 2) Ticker(t).history with long period, then slice
        # 3) shorter recent period as a last resort
        # Returns Series of Close indexed by date or None
        try:
            df = yf.download(t, start=start, end=end, interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # If multi-index (Columns like ('Close', t)), normalize
                if isinstance(df.columns, pd.MultiIndex):
                    if ("Close", t) in df.columns:
                        s = df[("Close", t)].rename(t)
                    elif ("Close", "") in df.columns:
                        s = df[("Close", "")].rename(t)
                    else:
                        s = df["Close"].rename(t) if "Close" in df.columns else None
                else:
                    s = df["Close"].rename(t) if "Close" in df.columns else None
                if s is not None and s.dropna().shape[0] > 0:
                    return s
        except Exception as _:
            pass

        try:
            s = None
            hist = yf.Ticker(t).history(period="20y", interval="1d", auto_adjust=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
                hist = hist.sort_index()
                # slice by date (inclusive of start, exclusive of end like yf)
                s = hist.loc[(hist.index >= pd.to_datetime(start)) & (hist.index < pd.to_datetime(end)), "Close"].rename(t)
                if s.dropna().shape[0] > 0:
                    return s
        except Exception as _:
            pass

        try:
            # last resort: try a smaller recent period
            hist = yf.Ticker(t).history(period="3mo", interval="1d", auto_adjust=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
                hist = hist.sort_index()
                s = hist.loc[(hist.index >= pd.to_datetime(start)) & (hist.index < pd.to_datetime(end)), "Close"].rename(t)
                if s.dropna().shape[0] > 0:
                    return s
        except Exception as _:
            pass

        return None

    series_list = []
    for t in tickers_list:
        # First try cache immediately (fast)
        cached = _from_cache(t)
        if cached is not None and not cached.empty:
            series_list.append(cached[t])
            continue

        last_err = None
        for k in range(retries):
            # Backoff with jitter
            delay = base_delay * (2 ** k) + random.uniform(0, 0.75)
            if k > 0:
                time.sleep(delay)

            try:
                s = _attempt_download(t, start, end)
                if s is not None and not s.empty:
                    # Save to cache and move on
                    df_close = s.to_frame(name=t)
                    _to_cache(t, df_close)
                    series_list.append(s)
                    break
            except Exception as e:
                last_err = e
                # keep retrying
        else:
            # If couldn't download after retries, try cache again (maybe partial older cache exists)
            cached = _from_cache(t)
            if cached is not None and not cached.empty:
                series_list.append(cached[t])
            else:
                # Create an empty series placeholder
                series_list.append(pd.Series(name=t, dtype=float))

    if len(series_list) == 0:
        raise Exception("Download failed and no cache available for requested tickers.")

    # Combine into DataFrame, align dates, drop rows that are all NaN
    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index < pd.to_datetime(end))]
    df = df.dropna(how="all")

    if df.empty:
        tickers_str = ", ".join(tickers_list)
        raise Exception(f"No data available after retries and cache for: {tickers_str}.")

    return df


def simulate_spxl_data(gspc_data, start_date, spxl_inception_date=None, spxl_inception_price=None):
    expense_ratio = 0.0087  # 0.87% annual expense ratio
    daily_expense = expense_ratio / 252
    leverage = 3
    simulated_prices = []
    prev_price = spxl_inception_price if spxl_inception_price else gspc_data.iloc[0]  # Start with ^GSPC price if no inception price

    gspc_returns = gspc_data.pct_change().fillna(0)
    
    for i in range(len(gspc_data)):
        daily_return = leverage * gspc_returns.iloc[i] - daily_expense
        prev_price = prev_price * (1 + daily_return)
        simulated_prices.append(prev_price)
    
    return pd.Series(simulated_prices, index=gspc_data.index, name='Close_SPXL')

def compute_strategy(data, strategy_name='HG_Strategy', use_simulated=False, include_expense=True):
    spxl_column = 'Close_SPXL_SIM' if use_simulated else 'Close_SPXL'
    data['SMA_80'] = data[spxl_column].rolling(window=62).mean()
    data['SMA_80_Slope'] = data['SMA_80'].diff()
    
    strategy_value = 1.0
    strategy_returns = []
    holding = True
    prev_price = data[spxl_column].iloc[0]
    buy_signals = []
    sell_signals = []
    expense_ratio = 0.1087  # 0.87% annual expense ratio
    daily_expense = expense_ratio / 252
    hold_start_dates = [None] * len(data)
    hold_prices = [None] * len(data)
    holding_days = [False] * len(data)

    for i in range(len(data)):
        price = data[spxl_column].iloc[i]
        slope = data['SMA_80_Slope'].iloc[i]
        if pd.isna(slope):
            strategy_returns.append(strategy_value)
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            hold_start_dates[i] = None
            hold_prices[i] = None
            holding_days[i] = False
            continue
        if holding and data['SMA_80_Slope'].iloc[i - 1] > 0 and slope <= 0:
            holding = False
            sell_signals.append(data['Close_SPX'].iloc[i])
            buy_signals.append(np.nan)
            hold_start_dates[i] = None
            hold_prices[i] = None
            holding_days[i] = False
        elif not holding and slope > 0:
            holding = True
            prev_price = price
            buy_signals.append(data['Close_SPX'].iloc[i])
            sell_signals.append(np.nan)
            hold_start_dates[i] = data.index[i]
            hold_prices[i] = price
            holding_days[i] = True
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            hold_start_dates[i] = hold_start_dates[i-1] if i > 0 else None
            hold_prices[i] = hold_prices[i-1] if holding else None
            holding_days[i] = holding
        if holding:
            daily_return = price / prev_price
            if include_expense:
                daily_return *= (1 - daily_expense)
            strategy_value *= daily_return
            prev_price = price
        strategy_returns.append(strategy_value)
    
    data[strategy_name] = strategy_returns
    if strategy_name == 'HG_Strategy':
        data['Buy_Signal'] = buy_signals
        data['Sell_Signal'] = sell_signals
        data['Hold_Start_Dates'] = hold_start_dates
        data['Hold_Prices'] = hold_prices
        data['Holding_Days'] = holding_days
    return data

def compute_cross_strategy(data, strategy_name='HG_Cross', use_simulated=False, include_expense=True):
    spxl_column = 'Close_SPXL_SIM' if use_simulated else 'Close_SPXL'
    data['SMA_35'] = data[spxl_column].rolling(window=35).mean()
    data['SMA_85'] = data[spxl_column].rolling(window=85).mean()
    
    strategy_value = 1.0
    strategy_returns = []
    holding = False
    prev_price = data[spxl_column].iloc[0]
    buy_signals = []
    sell_signals = []
    expense_ratio = 0.0087  # 0.87% annual expense ratio
    daily_expense = expense_ratio / 252
    hold_start_dates = [None] * len(data)
    hold_prices = [None] * len(data)
    holding_days = [False] * len(data)

    for i in range(len(data)):
        if i == 0:
            strategy_returns.append(strategy_value)
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            hold_start_dates[i] = None
            hold_prices[i] = None
            holding_days[i] = False
            continue
        
        price = data[spxl_column].iloc[i]
        sma_short = data['SMA_35'].iloc[i]
        sma_long = data['SMA_85'].iloc[i]
        prev_short = data['SMA_35'].iloc[i-1]
        prev_long = data['SMA_85'].iloc[i-1]
        
        if pd.isna(sma_short) or pd.isna(sma_long):
            strategy_returns.append(strategy_value)
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            hold_start_dates[i] = None
            hold_prices[i] = None
            holding_days[i] = False
            continue
        
        if not holding and prev_short <= prev_long and sma_short > sma_long:
            holding = True
            prev_price = price
            buy_signals.append(data['Close_SPX'].iloc[i])
            sell_signals.append(np.nan)
            hold_start_dates[i] = data.index[i]
            hold_prices[i] = price
            holding_days[i] = True
        elif holding and prev_short >= prev_long and sma_short < sma_long:
            holding = False
            sell_signals.append(data['Close_SPX'].iloc[i])
            buy_signals.append(np.nan)
            hold_start_dates[i] = None
            hold_prices[i] = None
            holding_days[i] = False
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            hold_start_dates[i] = hold_start_dates[i-1] if i > 0 else None
            hold_prices[i] = hold_prices[i-1] if holding else None
            holding_days[i] = holding
        if holding:
            daily_return = price / prev_price
            if include_expense:
                daily_return *= (1 - daily_expense)
            strategy_value *= daily_return
            prev_price = price
        strategy_returns.append(strategy_value)
    
    data[strategy_name] = strategy_returns
    data['Buy_Signal_Cross'] = buy_signals
    data['Sell_Signal_Cross'] = sell_signals
    data['Hold_Start_Dates_Cross'] = hold_start_dates
    data['Hold_Prices_Cross'] = hold_prices
    data['Holding_Days_Cross'] = holding_days
    return data

def calculate_metrics(data, start_date, selected):
    years = (data.index[-1] - start_date).days / 365.25
    metrics = {}
    
    for strategy in selected:
        if strategy in data.columns and not data[strategy].isna().all():
            # Calculate CAGR
            cagr = (data[strategy].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate Alpha (relative to SPX_BuyHold)
            alpha = data[strategy].iloc[-1] / data['SPX_BuyHold'].iloc[-1] if 'SPX_BuyHold' in data.columns else 1.0
            
            # Calculate Max Drawdown
            max_dd = ((data[strategy].cummax() - data[strategy]) / data[strategy].cummax()).max()
            
            # Calculate Time Invested
            if strategy in ['HG_Strategy', 'HG_SIM', 'HG_Cross']:
                if strategy == 'HG_Cross':
                    invested_days = data['Holding_Days_Cross'].sum() if 'Holding_Days_Cross' in data.columns else None
                else:
                    invested_days = data['Holding_Days'].sum() if 'Holding_Days' in data.columns else None
                if invested_days is not None and len(data) > 0:
                    time_invested = f"{invested_days / len(data):.1%}"
                else:
                    time_invested = "n/a"
            else:
                time_invested = "100%"
            
            # Expense Ratio
            expense_ratio = "0.87%" if strategy in ['HG_Strategy', 'HG_SIM', 'SPXL_BuyHold', 'HG_Cross'] else "0.0945%"
            
            # Return Multiple
            return_multiple = data[strategy].iloc[-1]
            
            # Calculate Number of Trades
            if strategy in ['HG_Strategy', 'HG_SIM', 'HG_Cross']:
                if strategy == 'HG_Cross':
                    num_trades = data['Buy_Signal_Cross'].count() if 'Buy_Signal_Cross' in data.columns else 0
                else:
                    num_trades = data['Buy_Signal'].count() if 'Buy_Signal' in data.columns else 0
            else:
                num_trades = 1  # Buy-and-hold strategies have 1 trade (initial purchase)
            
            # Store metrics
            metrics[strategy] = {
                'CAGR': f"{cagr:.2%}",
                'Alpha': f"{alpha:.2f}x",
                'Max Drawdown': f"{max_dd:.2%}",
                'Time Invested': time_invested,
                'Expense Ratio': expense_ratio,
                'Return Multiple': f"{return_multiple:.2f}x",
                'Number of Trades': f"{int(num_trades)}"
            }
    
    return metrics



def get_recent_trades_from_data(data, use_simulated=False, limit=10):
    """
    from pathlib import Path
    Build a list of the most recent buy/sell events for the primary Signal Hawk strategy.
    Returns a list of dicts: {'Date': 'YYYY-MM-DD', 'Action': 'Buy'/'Sell', 'Price': float, 'Return%': str or ''}
    """
    try:
        spxl_column = 'Close_SPXL_SIM' if use_simulated else 'Close_SPXL'
        if 'Buy_Signal' not in data.columns and 'Buy_Signal_Cross' in data.columns:
            buy_col = 'Buy_Signal_Cross'
            sell_col = 'Sell_Signal_Cross'
        else:
            buy_col = 'Buy_Signal'
            sell_col = 'Sell_Signal'

        rows = []
        last_buy_price = None

        for dt, row in data.iterrows():
            if buy_col in data.columns and pd.notna(row.get(buy_col)) and bool(row.get(buy_col)):
                price = float(row.get(spxl_column, float('nan')))
                rows.append({"Date": dt.strftime("%Y-%m-%d"), "Action": "Buy", "Price": price, "Return%": ""})
                last_buy_price = price
            if sell_col in data.columns and pd.notna(row.get(sell_col)) and bool(row.get(sell_col)):
                price = float(row.get(spxl_column, float('nan')))
                ret = ""
                if last_buy_price is not None:
                    try:
                        ret_val = (price / last_buy_price) - 1.0
                        ret = f"{ret_val:.2%}"
                    except Exception:
                        ret = ""
                rows.append({"Date": dt.strftime("%Y-%m-%d"), "Action": "Sell", "Price": price, "Return%": ret})

        # Sort by date desc and limit
        rows_sorted = sorted(rows, key=lambda r: r["Date"], reverse=True)[:limit]
        return rows_sorted
    except Exception as e:
        print(f"get_recent_trades_from_data error: {e}")
        return []


def apply_taxes(data, investment):
    try:
        taxed_hg = data['HG_Strategy'].copy()
        taxed_hg_sim = data['HG_SIM'].copy()
        taxed_spxl = data['SPXL_BuyHold'].copy()
        taxed_spx = data['SPX_BuyHold'].copy()
        taxed_hg_cross = data['HG_Cross'].copy()
        short_term_rate = 0.30
        long_term_rate = 0.20

        prev_hg = taxed_hg.iloc[0]
        prev_hg_sim = taxed_hg_sim.iloc[0]
        prev_spxl = taxed_spxl.iloc[0]
        prev_spx = taxed_spx.iloc[0]
        prev_hg_cross = taxed_hg_cross.iloc[0]
        hold_start_dates = data['Hold_Start_Dates']
        sell_signals = data['Sell_Signal']
        hold_start_dates_cross = data['Hold_Start_Dates_Cross']
        sell_signals_cross = data['Sell_Signal_Cross']

        for i in range(1, len(data)):
            current_hg = taxed_hg[i-1]
            current_hg_sim = taxed_hg_sim[i-1]
            current_spxl = taxed_spxl[i-1]
            current_spx = taxed_spx[i-1]
            current_hg_cross = taxed_hg_cross[i-1]
            new_hg = taxed_hg[i]
            new_hg_sim = taxed_hg_sim[i]
            new_spxl = taxed_spxl[i]
            new_spx = taxed_spx[i]
            new_hg_cross = taxed_hg_cross[i]

            if not pd.isna(sell_signals.iloc[i]) and hold_start_dates[i-1] is not None:
                hold_period = data.index[i] - hold_start_dates[i-1]
                days_held = hold_period.days
                tax_rate = long_term_rate if days_held >= 365 else short_term_rate
                gain_hg = (new_hg - current_hg) * investment
                gain_hg_sim = (new_hg_sim - current_hg_sim) * investment
                if gain_hg > 0:
                    tax_amount = gain_hg * tax_rate
                    taxed_hg[i] -= tax_amount / investment
                if gain_hg_sim > 0:
                    tax_amount = gain_hg_sim * tax_rate
                    taxed_hg_sim[i] -= tax_amount / investment
                prev_hg = new_hg
                prev_hg_sim = new_hg_sim

            if not pd.isna(sell_signals_cross.iloc[i]) and hold_start_dates_cross[i-1] is not None:
                hold_period = data.index[i] - hold_start_dates_cross[i-1]
                days_held = hold_period.days
                tax_rate = long_term_rate if days_held >= 365 else short_term_rate
                gain_hg_cross = (new_hg_cross - current_hg_cross) * investment
                if gain_hg_cross > 0:
                    tax_amount = gain_hg_cross * tax_rate
                    taxed_hg_cross[i] -= tax_amount / investment
                prev_hg_cross = new_hg_cross

            gain_spxl = (new_spxl - current_spxl) * investment
            gain_spx = (new_spx - current_spx) * investment
            if gain_spxl > 0:
                tax_rate_spxl = long_term_rate
                tax_amount_spxl = gain_spxl * tax_rate_spxl
                taxed_spxl[i] -= tax_amount_spxl / investment
            if gain_spx > 0:
                tax_rate_spx = long_term_rate
                tax_amount_spx = gain_spx * tax_rate_spx
                taxed_spx[i] -= tax_amount_spx / investment
            prev_spxl = new_spxl
            prev_spx = new_spx

        return taxed_hg, taxed_hg_sim, taxed_spxl, taxed_spx, taxed_hg_cross
    except Exception as e:
        print(f"Error in apply_taxes: {str(e)}")
        return data['HG_Strategy'], data['HG_SIM'], data['SPXL_BuyHold'], data['SPX_BuyHold'], data['HG_Cross']


# ==== Recent Trades helpers ====
def build_recent_trades_html_from_data(data, use_simulated=False, limit=30):
    """
    Build a paired BUY/SELL table for the Signal Hawk strategy using Buy_Signal and Sell_Signal columns.
    Shows Date, Action, SPXL Price, SPX Price, and Pair Return %. Most recent first, limited to `limit` rows.
    """
    try:
        spxl_col = 'Close_SPXL_SIM' if use_simulated else 'Close_SPXL'
        spx_col = 'Close_SPX'
        buy_col = 'Buy_Signal'
        sell_col = 'Sell_Signal'
        if buy_col not in data.columns or sell_col not in data.columns:
            return Markup("<div class='metrics'><h2>Recent Trades</h2><p style='text-align:center;'>No trades available.</p></div>")

        rows = []
        last_buy_price = None
        last_buy_date = None

        # Iterate chronologically
        for dt, row in data.iterrows():
            spxl_price = row.get(spxl_col, float('nan'))
            spx_price = row.get(spx_col, float('nan'))

            # BUY
            if pd.notna(row.get(buy_col)):
                rows.append({
                    "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "Action": "Buy",
                    "SPXL Price": float(spxl_price) if pd.notna(spxl_price) else None,
                    "SPX Price": float(spx_price) if pd.notna(spx_price) else None,
                    "Pair Return %": ""
                })
                last_buy_price = float(spxl_price) if pd.notna(spxl_price) else None
                last_buy_date = dt

            # SELL
            if pd.notna(row.get(sell_col)):
                pr = ""
                if last_buy_price is not None and pd.notna(spxl_price) and last_buy_price != 0:
                    try:
                        pr = f"{(float(spxl_price)/last_buy_price - 1.0)*100.0:.2f}"
                    except Exception:
                        pr = ""
                rows.append({
                    "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "Action": "Sell",
                    "SPXL Price": float(spxl_price) if pd.notna(spxl_price) else None,
                    "SPX Price": float(spx_price) if pd.notna(spx_price) else None,
                    "Pair Return %": pr
                })
                last_buy_price = None
                last_buy_date = None

        # If an open buy remains
        if last_buy_price is not None:
            # get most recent SPXL for "Open" return
            try:
                latest_spxl = float(data[spxl_col].dropna().iloc[-1])
                open_ret = (latest_spxl/last_buy_price - 1.0)*100.0
                # update the last Buy row's Pair Return % to show Open
                for i in range(len(rows)-1, -1, -1):
                    if rows[i]["Action"] == "Buy":
                        rows[i]["Pair Return %"] = f"Open ({open_ret:.2f}%)"
                        break
            except Exception:
                # leave as "Open" without percentage
                for i in range(len(rows)-1, -1, -1):
                    if rows[i]["Action"] == "Buy":
                        rows[i]["Pair Return %"] = "Open"
                        break

        # Sort by date desc and limit
        rows_sorted = sorted(rows, key=lambda r: r["Date"], reverse=True)[:limit]

        # Render HTML
        def fmt_price(x):
            if x is None or (isinstance(x,float) and (np.isnan(x))):
                return ""
            try:
                return f"${float(x):,.2f}"
            except Exception:
                return str(x)

        html = [
            "<div class='metrics'>",
            "<h2>Recent Trades (Signal Hawk)</h2>",
            "<table class='metrics-table'>",
            "<thead><tr><th>Date</th><th>Action</th><th>SPXL Price</th><th>SPX Price</th><th>Pair Return %</th></tr></thead>",
            "<tbody>"
        ]
        for r in rows_sorted:
            pr = r.get("Pair Return %", "")
            html.append(
                f"<tr><td>{r['Date']}</td><td>{r['Action']}</td><td>{fmt_price(r['SPXL Price'])}</td><td>{fmt_price(r['SPX Price'])}</td><td>{pr}</td></tr>"
            )
        html += ["</tbody></table></div>"]
        return Markup("".join(html))
    except Exception as e:
        print(f"build_recent_trades_html_from_data error: {e}")
        return Markup("<div class='metrics'><h2>Recent Trades</h2><p style='text-align:center;'>No trades available.</p></div>")

def build_all_trades_table(data_full, start_date, end_date, investment_amount=None, use_simulated=False):
    """
    Robust trade log table builder with carry-in handling.
    - Inserts a synthetic Buy* at start_date if a position was opened before start_date.
    - Uses investment_amount to compute whole-share quantity on each Buy.
    """
    import math
    import pandas as pd
    from markupsafe import Markup

    spxl_col = 'Close_SPXL_SIM' if use_simulated else 'Close_SPXL'
    spx_col  = 'Close_SPX'
    buy_col  = 'Buy_Signal'
    sell_col = 'Sell_Signal'

    for col in [spxl_col, spx_col, buy_col, sell_col]:
        if col not in data_full.columns:
            return Markup("<div class='metrics'><h2>All Trades (Signal Hawk)</h2><p style='text-align:center;'>No trades available.</p></div>")

    df = data_full.copy()

    # Determine carry-in
    in_position = False
    for _, row in df[df.index < start_date].iterrows():
        if pd.notna(row.get(buy_col)) and not in_position:
            in_position = True
        if pd.notna(row.get(sell_col)) and in_position:
            in_position = False

    rows = []
    buy_price = None
    buy_shares = None

    if in_position:
        start_row = df.loc[df.index >= start_date].iloc[0]
        buy_price = float(start_row[spxl_col])
        spx_px    = float(start_row[spx_col])
        if investment_amount and buy_price > 0:
            try:
                buy_shares = math.floor(float(investment_amount) / float(buy_price))
            except Exception:
                buy_shares = None
        rows.append({
            "Date": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            "Action": "Buy*",
            "SPXL Price": f"${buy_price:,.2f}",
            "SPX Price": f"{spx_px:,.2f}",
            "Shares": (str(int(buy_shares)) if buy_shares is not None else ""),
            "Pair Return %": ""
        })

    view = df[(df.index >= start_date) & (df.index <= end_date)]
    for dt, row in view.iterrows():
        spxl_px = float(row[spxl_col])
        spx_px  = float(row[spx_col])
        is_buy  = pd.notna(row.get(buy_col))
        is_sell = pd.notna(row.get(sell_col))

        if is_buy and buy_price is None:
            buy_price = spxl_px
            if investment_amount and buy_price > 0:
                try:
                    buy_shares = math.floor(float(investment_amount) / float(buy_price))
                except Exception:
                    buy_shares = None
            rows.append({
                "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                "Action": "Buy",
                "SPXL Price": f"${buy_price:,.2f}",
                "SPX Price": f"{spx_px:,.2f}",
                "Shares": (str(int(buy_shares)) if buy_shares is not None else ""),
                "Pair Return %": ""
            })
        elif is_sell and buy_price is not None:
            pair_ret = (spxl_px / buy_price - 1.0) * 100.0
            rows.append({
                "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                "Action": "Sell",
                "SPXL Price": f"${spxl_px:,.2f}",
                "SPX Price": f"{spx_px:,.2f}",
                "Shares": (str(int(buy_shares)) if buy_shares is not None else ""),
                "Pair Return %": f"{pair_ret:.2f}%"
            })
            buy_price = None
            buy_shares = None

    if buy_price is not None and len(view) > 0:
        last_px = float(view[spxl_col].iloc[-1])
        open_ret = (last_px / buy_price - 1.0) * 100.0
        for i in range(len(rows)-1, -1, -1):
            if rows[i]["Pair Return %"] == "":
                rows[i]["Pair Return %"] = f"Open ({open_ret:.2f}%)"
                break

    head = (
        "<div class='metrics'>"
        "<h2>All Trades (Signal Hawk)</h2>"
        "<table class='metrics-table'>"
        "<thead><tr>"
        "<th>Date</th><th>Action</th><th>SPXL Price</th><th>SPX Price</th><th>Shares</th><th>Pair Return %</th>"
        "</tr></thead><tbody>"
    )
    body_parts = []
    for r in reversed(rows):
        body_parts.append(
            f"<tr><td>{r['Date']}</td><td>{r['Action']}</td>"
            f"<td>{r['SPXL Price']}</td><td>{r['SPX Price']}</td>"
            f"<td>{r['Shares']}</td><td>{r['Pair Return %']}</td></tr>"
        )
    tail = "</tbody></table></div>"
    return Markup(head + "".join(body_parts) + tail)
# ==== End helpers ====

@app.route("/", methods=["GET", "POST"])
def index():
    investment = 10000
    start_date = datetime(2010, 1, 1)
    end_date = get_previous_trading_day_end()
    include_expense = True
    include_taxes = False
    options = [
        {'name': 'Signal Hawk Strategy', 'value': 'HG_Strategy'},
        {'name': 'HG_SIM Strategy', 'value': 'HG_SIM'},
        {'name': 'SPXL Buy & Hold', 'value': 'SPXL_BuyHold'},
        {'name': 'S&P 500 (SPX) Buy & Hold', 'value': 'SPX_BuyHold'},
        {'name': 'NVDA Buy & Hold', 'value': 'NVDA'},
        {'name': 'AMZN Buy & Hold', 'value': 'AMZN'},
        {'name': 'AAPL Buy & Hold', 'value': 'AAPL'},
        {'name': 'TSLA Buy & Hold', 'value': 'TSLA'},
        {'name': 'HG_Cross Strategy', 'value': 'HG_Cross'},
    ]
    default_selected = ['HG_Strategy', 'SPXL_BuyHold', 'SPX_BuyHold', 'HG_Cross']
    selected = default_selected

    try:
        if request.method == "POST":
            investment = float(request.form["investment"])
            start_date = datetime.strptime(request.form["start_date"], "%Y-%m-%d")
            include_expense = 'include_expense' in request.form
            include_taxes = 'include_taxes' in request.form
            selected = request.form.getlist("selected_options")

        # Ensure HG_Strategy is always computed for the summary
        if 'HG_Strategy' not in selected:
            selected.append('HG_Strategy')

        # Define tickers to fetch
        base_tickers = ['^GSPC', 'SPXL']
        additional_tickers = [s for s in selected if s in ['NVDA', 'AMZN', 'AAPL', 'TSLA']]
        all_tickers = base_tickers + additional_tickers

        # Download data with warmup buffer
        fetch_start = start_date - timedelta(days=400)
        raw_data = download_data(all_tickers, fetch_start, end_date)
        if raw_data.empty or '^GSPC' not in raw_data.columns:
            raise ValueError("No data available for the selected date range or tickers.")
        
        # Prepare SPXL data (use actual or simulated)
        spxl_inception_date = datetime(2008, 11, 5)
        spxl_data = raw_data['SPXL'] if 'SPXL' in raw_data.columns and not raw_data['SPXL'].isna().all() else pd.Series(index=raw_data.index)
        if start_date < spxl_inception_date:
            gspc_for_simulation = raw_data['^GSPC']  # Already fetched
            spxl_inception_price = yf.Ticker('SPXL').history(start=spxl_inception_date, end=spxl_inception_date + timedelta(days=1))['Close'].iloc[0]
            spxl_data = simulate_spxl_data(gspc_for_simulation, start_date, spxl_inception_date, spxl_inception_price)
            if 'SPXL' in raw_data.columns:
                spxl_data[spxl_inception_date:] = raw_data['SPXL'][spxl_inception_date:]

        # Create DataFrame
        data = pd.DataFrame({
            'Close_SPX': raw_data['^GSPC'],
            'Close_SPXL': spxl_data,
            'Close_SPXL_SIM': spxl_data
        })

        # Add additional assets to data
        for ticker in additional_tickers:
            if ticker in raw_data.columns:
                data[f'Close_{ticker}'] = raw_data[ticker]
        
        # Calculate buy-and-hold for base
        data['SPXL_BuyHold'] = data['Close_SPXL'] / data['Close_SPXL'].iloc[0]
        data['SPX_BuyHold'] = data['Close_SPX'] / data['Close_SPX'].iloc[0]

        # Calculate buy-and-hold for additional assets
        for ticker in additional_tickers:
            if f'Close_{ticker}' in data.columns and not data[f'Close_{ticker}'].isna().all():
                first_valid = data[f'Close_{ticker}'].first_valid_index()
                if first_valid is not None:
                    data[f'{ticker}_BuyHold'] = data[f'Close_{ticker}'] / data[f'Close_{ticker}'].loc[first_valid]
                else:
                    data[f'{ticker}_BuyHold'] = pd.Series(np.nan, index=data.index)
            else:
                data[f'{ticker}_BuyHold'] = pd.Series(np.nan, index=data.index)

        # Compute strategies
        data = compute_strategy(data, strategy_name='HG_Strategy', use_simulated=False, include_expense=include_expense)
        data = compute_strategy(data, strategy_name='HG_SIM', use_simulated=True, include_expense=include_expense)
        data = compute_cross_strategy(data, strategy_name='HG_Cross', use_simulated=False, include_expense=include_expense)
        # Now that signals are computed with full warmup, filter to user-selected range
        data_full = data.copy()
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if data.empty:
            raise ValueError("No data available for the selected date range.")

        # Re-base Buy & Hold from the first filtered bar
        data['SPXL_BuyHold'] = data['Close_SPXL'] / data['Close_SPXL'].iloc[0]
        data['SPX_BuyHold']  = data['Close_SPX'] / data['Close_SPX'].iloc[0]


        # Apply expense ratio to buy-and-hold if toggled
        if include_expense:
            data['SPXL_BuyHold'] *= (1 - 0.0087 / 252) ** ((data.index - data.index[0]).days / 365.25 * 252)
        else:
            data['SPXL_BuyHold'] = data['Close_SPXL'] / data['Close_SPXL'].iloc[0]  # Recalculate without expense

        # Apply taxes if toggled
        if include_taxes:
            taxed_hg, taxed_hg_sim, taxed_spxl, taxed_spx, taxed_hg_cross = apply_taxes(data, investment)
            data['HG_Strategy'] = taxed_hg
            data['HG_SIM'] = taxed_hg_sim
            data['SPXL_BuyHold'] = taxed_spxl
            data['SPX_BuyHold'] = taxed_spx
            data['HG_Cross'] = taxed_hg_cross

        # Create plot
        plot = go.Figure()
        trace_map = {
            'HG_Strategy': go.Scatter(x=data.index, y=data["HG_Strategy"] * investment, mode="lines", name="Signal Hawk Strategy"),
            'HG_SIM': go.Scatter(x=data.index, y=data["HG_SIM"] * investment, mode="lines", name="HG_SIM Strategy", line=dict(dash='dash')),
            'SPXL_BuyHold': go.Scatter(x=data.index, y=data['SPXL_BuyHold'] * investment, mode="lines", name="SPXL Buy & Hold"),
            'SPX_BuyHold': go.Scatter(x=data.index, y=data['SPX_BuyHold'] * investment, mode="lines", name="SPX Buy & Hold"),
            'NVDA': go.Scatter(x=data.index, y=data.get('NVDA_BuyHold', pd.Series(np.nan, index=data.index)) * investment, mode="lines", name="NVDA Buy & Hold"),
            'AMZN': go.Scatter(x=data.index, y=data.get('AMZN_BuyHold', pd.Series(np.nan, index=data.index)) * investment, mode="lines", name="AMZN Buy & Hold"),
            'AAPL': go.Scatter(x=data.index, y=data.get('AAPL_BuyHold', pd.Series(np.nan, index=data.index)) * investment, mode="lines", name="AAPL Buy & Hold"),
            'TSLA': go.Scatter(x=data.index, y=data.get('TSLA_BuyHold', pd.Series(np.nan, index=data.index)) * investment, mode="lines", name="TSLA Buy & Hold"),
            'HG_Cross': go.Scatter(x=data.index, y=data["HG_Cross"] * investment, mode="lines", name="HG_Cross Strategy"),
        }

        for s in selected:
            if s in trace_map:
                plot.add_trace(trace_map[s])

        for year in range(data.index[0].year + 1, data.index[-1].year):
            plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
        plot.update_layout(
            title="Investment Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=800
        )
        graph_html = pio.to_html(plot, full_html=False)

        # Calculate metrics with selected strategies
        metrics = calculate_metrics(data, start_date, selected)
        try:
            recent_trades_html = build_all_trades_table(data_full, start_date, end_date, investment_amount=investment, use_simulated=False)
        except Exception as e:
            print(f"Recent trades build error: {e}")
            recent_trades_html = Markup("<div class='metrics'><h2>Recent Trades</h2><p style='text-align:center;'>No trades available.</p></div>")
        summary = f"If you invested ${investment:,.2f} on {start_date.strftime('%m/%d/%Y')} and used the Signal Hawk buy & sell alerts, your account value would be ${data['HG_Strategy'].iloc[-1] * investment:,.2f} by {end_date.strftime('%m/%d/%Y')}.*"
    
        return render_template(
            "index.html",
            graph=graph_html,
            metrics=metrics,
            investment=investment,
            formatted_investment=f"{investment:,.2f}",
            formatted_investment_with_dollar=f"${investment:,.2f}",
            start_date=start_date.strftime("%Y-%m-%d"),
            summary=summary,
            include_expense=include_expense,
            include_taxes=include_taxes,
            options=options,
            selected=selected,
            recent_trades_html=recent_trades_html,
            recent_trades=get_recent_trades_from_data(data, use_simulated=False, limit=10)
)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return "Error: " + str(e), 500

        print(f"Error in index route: {str(e)}")
        return "Error: " + str(e), 500


@app.route("/signals", methods=["GET"])
def signals():
    try:
        start_date = datetime(2009, 1, 1)
        end_date = get_previous_trading_day_end()
        investment = 10000  # Define investment for message

        # Download data from Yahoo Finance
        tickers = ['^GSPC', 'SPXL']
        spxl_inception_date = datetime(2008, 11, 5)
        raw_data = download_data(tickers, start_date, end_date)
        if raw_data.empty or '^GSPC' not in raw_data.columns:
            raise ValueError("No data available for the selected date range or tickers.")
        
        # Prepare SPXL data (use actual or simulated)
        spxl_data = raw_data['SPXL'] if 'SPXL' in raw_data.columns and not raw_data['SPXL'].isna().all() else pd.Series(index=raw_data.index)
        if start_date < spxl_inception_date:
            gspc_for_simulation = download_data(['^GSPC'], start_date, end_date)
            spxl_inception_price = yf.Ticker('SPXL').history(start=spxl_inception_date, end=spxl_inception_date + timedelta(days=1))['Close'].iloc[0]
            spxl_data = simulate_spxl_data(gspc_for_simulation['^GSPC'], start_date, spxl_inception_date, spxl_inception_price)
            if 'SPXL' in raw_data.columns:
                spxl_data[spxl_inception_date:] = raw_data['SPXL'][spxl_inception_date:]

        # Create DataFrame
        data = pd.DataFrame({
            'Close_SPX': raw_data['^GSPC'],
            'Close_SPXL': spxl_data
        })
        data = compute_strategy(data, strategy_name='HG_Strategy', use_simulated=False)

        # Add SPX_BuyHold for metrics calculation
        data['SPX_BuyHold'] = data['Close_SPX'] / data['Close_SPX'].iloc[0]

        # Calculate metrics for HG_Strategy and SPX_BuyHold
        selected = ['HG_Strategy', 'SPX_BuyHold']
        metrics = calculate_metrics(data, start_date, selected)

        # Price action chart (SPX with Buy/Sell Signals)
        price_plot = go.Figure()
        price_plot.add_trace(go.Scatter(x=data.index, y=data["Close_SPX"], mode="lines", name="SPX Price"))
        price_plot.add_trace(go.Scatter(
            x=data.index,
            y=data["Buy_Signal"],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", size=10, color="green")
        ))
        price_plot.add_trace(go.Scatter(
            x=data.index,
            y=data["Sell_Signal"],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", size=10, color="red")
        ))
        for year in range(data.index[0].year + 1, data.index[-1].year):
            price_plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
        price_plot.update_layout(
            title="SPX Price with Buy/Sell Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=800
        )
        price_graph = pio.to_html(price_plot, full_html=False)

        # SMA chart (80-day SMA for SPX and SPXL)
        sma_plot = go.Figure()
        sma_plot.add_trace(go.Scatter(x=data.index, y=data["SMA_80"], mode="lines", name="SPX 80-day SMA", line=dict(color="orange")))
        sma_plot.add_trace(go.Scatter(x=data.index, y=data["Close_SPXL"].rolling(window=80).mean(), mode="lines", name="SPXL 80-day SMA", line=dict(color="purple")))
        for year in range(data.index[0].year + 1, data.index[-1].year):
            sma_plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
        sma_plot.update_layout(
            title="80-day SMA for SPX and SPXL",
            xaxis_title="Date",
            yaxis_title="SMA Value ($)",
            template="plotly_dark",
            height=800
        )
        sma_graph = pio.to_html(sma_plot, full_html=False)

        # SPXL price chart with Buy/Sell Signals
        spxl_plot = go.Figure()
        spxl_plot.add_trace(go.Scatter(x=data.index, y=data["Close_SPXL"], mode="lines", name="SPXL Price"))
        spxl_plot.add_trace(go.Scatter(
            x=data.index,
            y=data["Buy_Signal"] * 0.9,  # Offset buy signals slightly below SPXL price for visibility
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", size=10, color="green")
        ))
        spxl_plot.add_trace(go.Scatter(
            x=data.index,
            y=data["Sell_Signal"] * 1.1,  # Offset sell signals slightly above SPXL price for visibility
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", size=10, color="red")
        ))
        for year in range(data.index[0].year + 1, data.index[-1].year):
            spxl_plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
        spxl_plot.update_layout(
            title="SPXL Price with Buy/Sell Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=800
        )
        spxl_graph = pio.to_html(spxl_plot, full_html=False)

        message = f"If you invested ${investment:,.2f} on {start_date.strftime('%m/%d/%Y')} and used the Signal Hawk buy & sell alerts your account value would be ${data['HG_Strategy'].iloc[-1] * investment:,.2f} by {end_date.strftime('%m/%d/%Y')}.*"

        return render_template("spx_price_sma_signals_page.html", price_graph=price_graph, sma_graph=sma_graph, spxl_graph=spxl_graph, message=message, metrics=metrics)
    except Exception as e:
        print(f"Error in signals route: {str(e)}")
        return "Error: " + str(e), 500

@app.route("/historical", methods=["GET", "POST"])
def historical():
    try:
        # Default values for custom period
        custom_start_date = datetime(2000, 1, 1)
        custom_end_date = get_previous_trading_day_end()
        custom_investment = 10000
        custom_error = None

        # Handle form submission
        if request.method == "POST":
            try:
                custom_start_date = datetime.strptime(request.form["custom_start_date"], "%Y-%m-%d")
                custom_end_date = datetime.strptime(request.form["custom_end_date"], "%Y-%m-%d")
                custom_investment = float(request.form["custom_investment"])
                if custom_start_date >= custom_end_date:
                    custom_error = "End date must be after start date."
                if custom_investment <= 0:
                    custom_error = "Investment amount must be positive."
            except ValueError:
                custom_error = "Invalid date or investment amount format."

        # Define fixed periods
        periods = [
            (datetime(1937, 1, 1), datetime(1947, 12, 31), "1937-1947"),
            (datetime(1965, 1, 1), datetime(1982, 12, 31), "1965-1982")
        ]
        plots = []

        # Process fixed periods
        for start_date, end_date, period_name in periods:
            raw_data = download_data(['^GSPC'], start_date, end_date)
            if raw_data.empty or '^GSPC' not in raw_data.columns:
                plots.append({
                    'title': f"HG_SIM Strategy ({period_name})",
                    'graph': f"<p>No data available for {period_name}</p>"
                })
                continue
            
            spxl_data = simulate_spxl_data(raw_data['^GSPC'], start_date)
            data = pd.DataFrame({
                'Close_SPX': raw_data['^GSPC'],
                'Close_SPXL_SIM': spxl_data
            })
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            if data.empty:
                plots.append({
                    'title': f"HG_SIM Strategy ({period_name})",
                    'graph': f"<p>No data available for {period_name}</p>"
                })
                continue

            data['SPX_BuyHold'] = data['Close_SPX'] / data['Close_SPX'].iloc[0]
            data = compute_strategy(data, strategy_name='HG_SIM', use_simulated=True)

            # Calculate metrics for HG_SIM and SPX_BuyHold
            selected = ['HG_SIM', 'SPX_BuyHold']
            metrics = calculate_metrics(data, start_date, selected)

            plot = go.Figure()
            plot.add_trace(go.Scatter(x=data.index, y=data["HG_SIM"] * 10000, mode="lines", name="HG_SIM Strategy", line=dict(dash='dash')))
            plot.add_trace(go.Scatter(x=data.index, y=data['SPX_BuyHold'] * 10000, mode="lines", name="SPX Buy & Hold"))
            for year in range(data.index[0].year + 1, data.index[-1].year):
                plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
            plot.update_layout(
                title=f"HG_SIM Strategy vs SPX Buy & Hold ({period_name})",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark",
                height=800
            )
            plots.append({
                'title': f"HG_SIM Strategy ({period_name})",
                'graph': pio.to_html(plot, full_html=False),
                'metrics': metrics
            })

        # Process custom period
        custom_plot = None
        custom_metrics = None
        if not custom_error:
            raw_data = download_data(['^GSPC'], custom_start_date, custom_end_date)
            if raw_data.empty or '^GSPC' not in raw_data.columns:
                custom_plot = f"<p>No data available for {custom_start_date.strftime('%Y-%m-%d')} to {custom_end_date.strftime('%Y-%m-%d')}</p>"
            else:
                spxl_data = simulate_spxl_data(raw_data['^GSPC'], custom_start_date)
                data = pd.DataFrame({
                    'Close_SPX': raw_data['^GSPC'],
                    'Close_SPXL_SIM': spxl_data
                })
                data = data[(data.index >= custom_start_date) & (data.index <= custom_end_date)]
                if data.empty:
                    custom_plot = f"<p>No data available for {custom_start_date.strftime('%Y-%m-%d')} to {custom_end_date.strftime('%Y-%m-%d')}</p>"
                else:
                    data['SPX_BuyHold'] = data['Close_SPX'] / data['Close_SPX'].iloc[0]
                    data = compute_strategy(data, strategy_name='HG_SIM', use_simulated=True)

                    # Calculate metrics for HG_SIM and SPX_BuyHold
                    selected = ['HG_SIM', 'SPX_BuyHold']
                    custom_metrics = calculate_metrics(data, custom_start_date, selected)

                    plot = go.Figure()
                    plot.add_trace(go.Scatter(x=data.index, y=data["HG_SIM"] * custom_investment, mode="lines", name="HG_SIM Strategy", line=dict(dash='dash')))
                    plot.add_trace(go.Scatter(x=data.index, y=data['SPX_BuyHold'] * custom_investment, mode="lines", name="SPX Buy & Hold"))
                    for year in range(data.index[0].year + 1, data.index[-1].year):
                        plot.add_vline(x=pd.Timestamp(f"{year}-01-01"), line=dict(color="gray", dash="dot", width=1), opacity=0.2)
                    plot.update_layout(
                        title=f"HG_SIM Strategy vs SPX Buy & Hold ({custom_start_date.strftime('%Y-%m-%d')} to {custom_end_date.strftime('%Y-%m-%d')})",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        template="plotly_dark",
                        height=800
                    )
                    custom_plot = pio.to_html(plot, full_html=False)

        plots.append({
            'title': f"Custom Period ({custom_start_date.strftime('%Y-%m-%d')} to {custom_end_date.strftime('%Y-%m-%d')})",
            'graph': custom_plot,
            'metrics': custom_metrics
        })

        recent_trades = get_recent_trades_from_data(data, use_simulated=False, limit=10)
        return render_template("historical.html",
                             plots=plots,
                             custom_start_date=custom_start_date.strftime("%Y-%m-%d"),
                             custom_end_date=custom_end_date.strftime("%Y-%m-%d"),
                             custom_investment=custom_investment,
                             custom_error=custom_error,
                             recent_trades=recent_trades)
    except Exception as e:
        print(f"Error in historical route: {str(e)}")
        return "Error: " + str(e), 500

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    try:
        if request.method == "POST":
            email = request.form["email"]
            username = request.form["username"]
            plan = request.form["plan"]
            return redirect(url_for("index"))
        return render_template("signup.html")
    except Exception as e:
        print(f"Error in signup route: {str(e)}")
        return "Error: " + str(e), 500

# --- Billing Portal ---
@app.post("/billing-portal")
def billing_portal():
    if not stripe.api_key:
        return ("Stripe not configured.", 500)
    user = session.get("user") or {}
    email = user.get("email")
    if not email:
        return ("No logged in user email found. Sign in first.", 400)
    custs = stripe.Customer.list(email=email, limit=1).data
    if not custs:
        return ("No Stripe customer found for this email. Complete checkout first.", 400)
    bp_sess = stripe.billing_portal.Session.create(
        customer=custs[0].id,
        return_url=f"{APP_BASE_URL}/account"
    )
    return jsonify({"url": bp_sess.url})

# --- Contact details update (updates Stripe Customer by current session email) ---
@app.post("/update-contact")
def update_contact():
    if not stripe.api_key:
        return ("Stripe not configured.", 500)
    user = session.get("user") or {}
    current_email = user.get("email")
    if not current_email:
        return ("No logged in user email found. Sign in first.", 400)

    data = request.get_json(force=True)
    new_email = (data.get("email") or "").strip()
    new_phone = (data.get("phone") or "").strip()

    custs = stripe.Customer.list(email=current_email, limit=1).data
    if not custs:
        return ("No Stripe customer found for this user.", 400)
    cust = custs[0]
    updated = stripe.Customer.modify(cust.id, email=new_email or None, phone=new_phone or None)

    # Update session user
    if new_email:
        session["user"]["email"] = new_email
    return jsonify({"ok": True})

# --- Preferences (stored in session for demo; persist to DB in production) ---
@app.post("/prefs")
def update_prefs():
    data = request.get_json(force=True)
    session["prefs"] = data
    return jsonify({"ok": True, "prefs": session["prefs"]})

if __name__ == "__main__":
    print("App initialized")
    app.run(debug=True)


# --- Enhanced Recent Trades renderer with Net Return footer (includes open trade) ---
def render_recent_trades_html(trades_df, current_spxl=None):
    """
    Render a HTML table for the 30 most recent trade rows.
    If a pair is still open (no SELL row), annotate the open line with current return.
    Adds a footer row showing the Net Return from the last 30 trades (pairs) including open.
    """
    import math, re
    import numpy as np
    import pandas as pd
    from markupsafe import Markup

    if trades_df is None or len(trades_df) == 0:
        return Markup("<div class='metrics'><h2>Recent Trades</h2><p style='text-align:center;'>No trades available.</p></div>")

    # For open trades, compute current return using latest SPXL if provided
    if current_spxl is not None:
        # Find any _pair_id that has a Buy but no Sell
        open_ids = set(trades_df.loc[trades_df["Action"]=="Buy","_pair_id"]) - set(trades_df.loc[trades_df["Action"]=="Sell","_pair_id"])
        for pid in open_ids:
            buy_px_series = trades_df.loc[(trades_df["_pair_id"]==pid) & (trades_df["Action"]=="Buy"), "SPXL Price"]
            if len(buy_px_series) > 0:
                buy_px = buy_px_series.iloc[-1]
                if buy_px and not math.isnan(buy_px) and float(buy_px) != 0.0:
                    curr_ret = (float(current_spxl) / float(buy_px) - 1.0) * 100.0
                    trades_df.loc[(trades_df["_pair_id"]==pid) & (trades_df["Action"]=="Buy"), "Pair Return %"] = f"Open ({curr_ret:.2f}%)"

    # Sort by Date descending, then ensure SELL shown after its BUY for same day if equal dates
    _dt = pd.to_datetime(trades_df["Date"], errors="coerce")
    trades_df = trades_df.assign(_dt=_dt).sort_values(["_dt","_side"], ascending=[False, True]).drop(columns=["_dt"])

    # Keep visible columns, but retain _pair_id for footer calc
    show = trades_df[["Date","Action","SPXL Price","SPX Price","Pair Return %","_pair_id"]].copy()
    show = show.head(30)  # 30 most recent rows

    # Format prices for display
    def _fmt_px(x):
        import pandas as pd, numpy as np
        try:
            if isinstance(x, str):
                # allow numeric strings
                xf = float(x)
                return f"${xf:,.2f}"
            return ("" if pd.isna(x) else f"${float(x):,.2f}") if isinstance(x,(int,float,np.floating)) else (x if x is not None else "")
        except Exception:
            return x if x is not None else ""
    show["SPXL Price"] = show["SPXL Price"].apply(_fmt_px)
    show["SPX Price"]  = show["SPX Price"].apply(_fmt_px)

    # Compute Net Return across pairs visible in the last 30 rows.
    # Rule: for each _pair_id in 'show', use the SELL row's 'Pair Return %' if present; otherwise, if BUY row has "Open (x%)", use x.
    # Ignore pairs with no numeric return yet.
    def _parse_pct(val):
        if val is None:
            return float('nan')
        if isinstance(val,(int,float)):
            return float(val)
        s = str(val)
        m = re.search(r'(-?\d+(?:\.\d+)?)', s)
        return float(m.group(1)) if m else float('nan')

    net_returns = []
    pairs_included = 0
    for pid in show["_pair_id"].unique():
        sub = show[show["_pair_id"]==pid]
        # Prefer SELL row value
        sell_row = sub[sub["Action"].str.lower()=="sell"]
        val = None
        if len(sell_row) > 0:
            val = _parse_pct(sell_row["Pair Return %"].iloc[-1])
        else:
            # look for BUY row with Open(x%)
            buy_row = sub[sub["Action"].str.lower()=="buy"]
            if len(buy_row) > 0:
                val = _parse_pct(buy_row["Pair Return %"].iloc[-1])
        if val is not None and not (pd.isna(val)):
            net_returns.append(val)
            pairs_included += 1

    net_return_sum = sum(net_returns) if len(net_returns) > 0 else 0.0
    net_return_sum_str = f"{net_return_sum:.2f}%"
    pairs_label = f"{pairs_included} trade{'s' if pairs_included != 1 else ''}"

    # Build HTML
    html = [
        "<div class='metrics'>",
        "<h2>Recent Trades (Signal Hawk)</h2>",
        "<table class='metrics-table'>",
        "<thead><tr><th>Date</th><th>Action</th><th>SPXL Price</th><th>SPX Price</th><th>Pair Return %</th></tr></thead>",
        "<tbody>",
    ]
    for _, r in show.iterrows():
        pr = "" if pd.isna(r.get("Pair Return %")) else r.get("Pair Return %")
        # If it’s still open and not computed above, label explicitly
        if (not pr) and (str(r["Action"]).lower() == "buy"):
            pr = "Open"
        html.append(
            f"<tr><td>{r['Date']}</td><td>{r['Action']}</td><td>{r['SPXL Price']}</td><td>{r['SPX Price']}</td><td>{pr}</td></tr>"
        )
    # Footer with Net Return
    html += [
        "</tbody>",
        f"<tfoot><tr><td colspan='5' style='text-align:right;font-weight:600;'>Net Return (last {pairs_label} in table): {net_return_sum_str}</td></tr></tfoot>",
        "</table>",
        "</div>"
    ]
    return Markup("".join(html))


# --- Improved Recent Trades renderer with Net Return summary row inside <tbody> ---
def render_recent_trades_html(trades_df, current_spxl=None):
    """
    Render a HTML table for the 30 most recent trade rows.
    If a pair is still open (no SELL row), annotate the open line with current return.
    Adds a final summary row *inside tbody* showing the Net Return from the last 30 trades (pairs), including any open trade.
    """
    import math, re
    import numpy as np
    import pandas as pd
    from markupsafe import Markup

    if trades_df is None or len(trades_df) == 0:
        return Markup("<div class='metrics'><h2>Recent Trades</h2><p style='text-align:center;'>No trades available.</p></div>")

    # For open trades, compute current return using latest SPXL if provided
    if current_spxl is not None and "SPXL Price" in trades_df.columns:
        open_ids = set(trades_df.loc[trades_df["Action"]=="Buy","_pair_id"]) - set(trades_df.loc[trades_df["Action"]=="Sell","_pair_id"])
        for pid in open_ids:
            buy_px_series = trades_df.loc[(trades_df["_pair_id"]==pid) & (trades_df["Action"]=="Buy"), "SPXL Price"]
            if len(buy_px_series) > 0:
                buy_px = buy_px_series.iloc[-1]
                try:
                    if buy_px and not math.isnan(float(buy_px)) and float(buy_px) != 0.0:
                        curr_ret = (float(current_spxl) / float(buy_px) - 1.0) * 100.0
                        trades_df.loc[(trades_df["_pair_id"]==pid) & (trades_df["Action"]=="Buy"), "Pair Return %"] = f"Open ({curr_ret:.2f}%)"
                except Exception:
                    pass

    # Sort by Date descending, then ensure SELL shown after its BUY for same day if equal dates
    _dt = pd.to_datetime(trades_df["Date"], errors="coerce")
    trades_df = trades_df.assign(_dt=_dt).sort_values(["_dt","_side"], ascending=[False, True]).drop(columns=["_dt"])

    # Keep visible columns, but retain _pair_id for footer calc
    cols_needed = ["Date","Action","SPXL Price","SPX Price","Pair Return %","_pair_id"]
    for c in cols_needed:
        if c not in trades_df.columns:
            trades_df[c] = np.nan
    show = trades_df[cols_needed].copy()
    show = show.head(30)  # 30 most recent rows

    # Format prices for display
    def _fmt_px(x):
        import pandas as pd, numpy as np
        try:
            if isinstance(x, str):
                xf = float(x)
                return f"${xf:,.2f}"
            return ("" if pd.isna(x) else f"${float(x):,.2f}") if isinstance(x,(int,float,np.floating)) else (x if x is not None else "")
        except Exception:
            return x if x is not None else ""
    show["SPXL Price"] = show["SPXL Price"].apply(_fmt_px)
    show["SPX Price"]  = show["SPX Price"].apply(_fmt_px)

    # Compute Net Return across unique pairs represented in these 30 rows.
    def _parse_pct(val):
        if val is None:
            return float('nan')
        if isinstance(val,(int,float)):
            return float(val)
        s = str(val)
        m = re.search(r'(-?\d+(?:\.\d+)?)', s)
        return float(m.group(1)) if m else float('nan')

    net_returns = []
    pairs_included = 0
    for pid in show["_pair_id"].dropna().unique():
        sub = show[show["_pair_id"]==pid]
        # Prefer SELL row value
        sell_row = sub[sub["Action"].astype(str).str.lower()=="sell"]
        val = None
        if len(sell_row) > 0:
            val = _parse_pct(sell_row["Pair Return %"].iloc[-1])
        else:
            buy_row = sub[sub["Action"].astype(str).str.lower()=="buy"]
            if len(buy_row) > 0:
                val = _parse_pct(buy_row["Pair Return %"].iloc[-1])
        if val is not None and not (pd.isna(val)):
            net_returns.append(val)
            pairs_included += 1

    net_return_sum = sum(net_returns) if len(net_returns) > 0 else 0.0
    net_return_sum_str = f"{net_return_sum:.2f}%"

    # Build HTML with summary row INSIDE tbody
    html = [
        "<div class='metrics'>",
        "<h2>Recent Trades (Signal Hawk)</h2>",
        "<table class='metrics-table'>",
        "<thead><tr><th>Date</th><th>Action</th><th>SPXL Price</th><th>SPX Price</th><th>Pair Return %</th></tr></thead>",
        "<tbody>",
    ]
    for _, r in show.iterrows():
        pr = "" if pd.isna(r.get("Pair Return %")) else r.get("Pair Return %")
        if (not pr) and (str(r["Action"]).lower() == "buy"):
            pr = "Open"
        html.append(
            f"<tr><td>{r['Date']}</td><td>{r['Action']}</td><td>{r['SPXL Price']}</td><td>{r['SPX Price']}</td><td>{pr}</td></tr>"
        )

    # Summary row
    html.append(f"<tr><td colspan='5' style='text-align:right;font-weight:600;'>Net Return (last {pairs_included} trade{'s' if pairs_included != 1 else ''} in table): {net_return_sum_str}</td></tr>")

    html += [
        "</tbody>",
        "</table>",
        "</div>"
    ]
    return Markup("".join(html))



# =======================

# =======================
# Signal Hawk: Payments + Users (SQLite) + Email Account Setup + Password Reset
# =======================
import os
import sqlite3
import secrets
import smtplib
from email.message import EmailMessage
from contextlib import closing
from datetime import datetime, timedelta
from functools import wraps

from flask import session, jsonify, abort

from werkzeug.security import generate_password_hash, check_password_hash

# Third-party libs
try:
    import stripe
except Exception:
    stripe = None
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# ---- Configuration (via env vars) ----
APP_SECRET_KEY        = os.environ.get("APP_SECRET_KEY", os.environ.get("FLASK_SECRET_KEY", "dev-change-me"))
DB_PATH               = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "signal_hawk.db"))

STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_MONTHLY  = os.environ.get("STRIPE_PRICE_MONTHLY", "")
STRIPE_PRICE_YEARLY   = os.environ.get("STRIPE_PRICE_YEARLY", "")
STRIPE_PRICE_LIFETIME = os.environ.get("STRIPE_PRICE_LIFETIME", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Email (SMTP) — recommended: SendGrid SMTP or Google Workspace SMTP
SMTP_HOST             = os.environ.get("SMTP_HOST", "")
SMTP_PORT             = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER             = os.environ.get("SMTP_USER", "")
SMTP_PASS             = os.environ.get("SMTP_PASS", "")
FROM_EMAIL            = os.environ.get("FROM_EMAIL", "support@signalhawkcapital.com")
SUPPORT_EMAIL         = os.environ.get("SUPPORT_EMAIL", "support@signalhawkcapital.com")
BASE_URL              = os.environ.get("BASE_URL", "")  # e.g. https://signalhawkcapital.com (optional but recommended)

# Twilio (optional)
TWILIO_SID            = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN          = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM           = os.environ.get("TWILIO_FROM_NUMBER", "")

# Flask session secret
app.secret_key = APP_SECRET_KEY

if stripe and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

def _external_url(path: str) -> str:
    """
    Build an absolute URL for emails. Prefer BASE_URL in production.
    Falls back to url_for(..., _external=True) when BASE_URL not set.
    """
    if BASE_URL:
        return BASE_URL.rstrip("/") + path
    # fallback: relies on request context; caller should pass url_for when possible
    return path

# ---------- Database helpers ----------
def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_users_columns():
    """Add new columns safely (idempotent) for auth + tokens."""
    needed = {
        "password_hash": "TEXT",
        "setup_token": "TEXT",
        "setup_token_expires": "TEXT",
        "reset_token": "TEXT",
        "reset_token_expires": "TEXT",
        "terms_accepted_at": "TEXT",
        "terms_accepted_ip": "TEXT",
        "terms_version": "TEXT",
    }
    with closing(db_connect()) as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        for name, ctype in needed.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {name} {ctype}")
        conn.commit()

def init_users_table():
    with closing(db_connect()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                phone TEXT,
                plan TEXT,
                stripe_customer_id TEXT,
                subscription_status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
    ensure_users_columns()

init_users_table()

def upsert_user(email:str, phone:str=None, plan:str=None, stripe_customer_id:str=None,
                status:str=None, password_hash:str=None,
                setup_token:str=None, setup_token_expires:str=None,
                reset_token:str=None, reset_token_expires:str=None,
                terms_accepted_at:str=None, terms_accepted_ip:str=None, terms_version:str=None):
    now = datetime.utcnow().isoformat()
    with closing(db_connect()) as conn:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            conn.execute("""
                UPDATE users SET
                    phone=COALESCE(?, phone),
                    plan=COALESCE(?, plan),
                    stripe_customer_id=COALESCE(?, stripe_customer_id),
                    subscription_status=COALESCE(?, subscription_status),
                    password_hash=COALESCE(?, password_hash),
                    setup_token=COALESCE(?, setup_token),
                    setup_token_expires=COALESCE(?, setup_token_expires),
                    reset_token=COALESCE(?, reset_token),
                    reset_token_expires=COALESCE(?, reset_token_expires),
                    terms_accepted_at=COALESCE(?, terms_accepted_at),
                    terms_accepted_ip=COALESCE(?, terms_accepted_ip),
                    terms_version=COALESCE(?, terms_version),
                    updated_at=?
                WHERE email=?
            """, (phone, plan, stripe_customer_id, status, password_hash,
                  setup_token, setup_token_expires, reset_token, reset_token_expires,
                  terms_accepted_at, terms_accepted_ip, terms_version,
                  now, email))
        else:
            conn.execute("""
                INSERT INTO users (email, phone, plan, stripe_customer_id, subscription_status,
                                   password_hash, setup_token, setup_token_expires,
                                   reset_token, reset_token_expires,
                                   terms_accepted_at, terms_accepted_ip, terms_version,
                                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (email, phone, plan, stripe_customer_id, status or "initiated",
                  password_hash, setup_token, setup_token_expires,
                  reset_token, reset_token_expires,
                  terms_accepted_at, terms_accepted_ip, terms_version,
                  now, now))
        conn.commit()

def get_user_by_email(email:str):
    with closing(db_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

def get_user_by_setup_token(token:str):
    with closing(db_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE setup_token=?", (token,)).fetchone()

def get_user_by_reset_token(token:str):
    with closing(db_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE reset_token=?", (token,)).fetchone()

def mark_setup_token(email:str, token:str, expires_at:str):
    upsert_user(email=email, setup_token=token, setup_token_expires=expires_at)

def clear_setup_token(email:str):
    upsert_user(email=email, setup_token="", setup_token_expires="")

def mark_reset_token(email:str, token:str, expires_at:str):
    upsert_user(email=email, reset_token=token, reset_token_expires=expires_at)

def clear_reset_token(email:str):
    upsert_user(email=email, reset_token="", reset_token_expires="")

def set_password(email:str, new_password:str):
    ph = generate_password_hash(new_password)
    upsert_user(email=email, password_hash=ph)

# ---------- Email helpers ----------
def send_email(to_email: str, subject: str, body_text: str) -> tuple[bool, str]:
    """
    Sends a plain-text email via SMTP.
    Configure SMTP_* and FROM_EMAIL in your .env.
    """
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and FROM_EMAIL):
        return False, "SMTP not configured (set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/FROM_EMAIL)"
    try:
        msg = EmailMessage()
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body_text)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)

def build_setup_email(to_email: str, setup_link: str) -> tuple[str, str]:
    subject = "Welcome to Signal Hawk Capital — Set Up Your Account"
    body = f"""Hi,

Welcome to Signal Hawk Capital — your subscription is now active.

To access your account, alerts, and analytics, please complete your account setup using the link below:

Set up your account: {setup_link}

This will allow you to:
- Log in securely
- Manage your subscription and billing
- Access Signal Hawk tools and resources

Important Notes
- Signal Hawk Capital, LLC provides educational market insights only
- We do not place trades or manage funds on your behalf
- You remain in full control of all investment decisions

If you did not create this account or have questions, contact us at {SUPPORT_EMAIL}.

Best regards,
Signal Hawk Capital, LLC

Educational use only. Investing involves risk, including loss of principal.
"""
    return subject, body

def build_reset_email(to_email: str, reset_link: str) -> tuple[str, str]:
    subject = "Signal Hawk Capital — Reset your password"
    body = f"""Hi,

We received a request to reset the password for your Signal Hawk Capital account.

Reset your password using this link (expires in 24 hours):
{reset_link}

If you did not request this, you can ignore this email.

Support: {SUPPORT_EMAIL}

Signal Hawk Capital, LLC
"""
    return subject, body

# ---------- Auth/session helpers ----------
def get_current_user():
    email = session.get("user_email")
    if not email:
        return None
    return get_user_by_email(email)

def login_user(email:str):
    session["user_email"] = email

def logout_user():
    session.pop("user_email", None)

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def subscription_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        if (user.get("subscription_status") or "").lower() != "active":
            return redirect(url_for("signup"))
        return fn(*args, **kwargs)
    return wrapper

# ---------- Twilio helper ----------
def send_sms(to:str, body:str):
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and TwilioClient):
        return False, "Twilio not configured"
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(to=to, from_=TWILIO_FROM, body=body)
        return True, msg.sid
    except Exception as e:
        return False, str(e)

# ---------- Stripe Checkout: create session ----------
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if not stripe:
        return ("Stripe SDK not installed. Run: pip install stripe", 500)
    if not STRIPE_SECRET_KEY:
        return ("STRIPE_SECRET_KEY not set", 500)

    data = request.get_json(force=True) if request.data else {}
    email = (data.get("email") or "").strip().lower()
    phone = (data.get("phone") or "").strip()
    plan = (data.get("plan") or "monthly").strip().lower()
    investor_type = (data.get("investorType") or "individual").strip().lower()

    accepted_terms = bool(data.get("accepted_terms"))
    if not accepted_terms:
        return ("You must accept the Terms/Disclosures/Privacy Policy to continue.", 400)

    # map plan -> price id
    price_id = STRIPE_PRICE_MONTHLY
    if plan == "yearly":
        price_id = STRIPE_PRICE_YEARLY
    elif plan == "lifetime":
        price_id = STRIPE_PRICE_LIFETIME

    if not price_id:
        return ("Stripe price ID not configured for selected plan", 500)

    # Store acceptance + initiated user record
    terms_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or ""
    terms_at = datetime.utcnow().isoformat()
    terms_version = os.environ.get("TERMS_VERSION", "v1")

    try:
        upsert_user(
            email=email,
            phone=phone,
            plan=f"{investor_type}:{plan}",
            stripe_customer_id=None,
            status="initiated",
            terms_accepted_at=terms_at,
            terms_accepted_ip=terms_ip,
            terms_version=terms_version
        )
    except Exception:
        pass

    try:
        sess = stripe.checkout.Session.create(
            mode="payment" if plan == "lifetime" else "subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=email,
            phone_number_collection={"enabled": True},
            metadata={
                "alerts_email": email,
                "alerts_phone": phone,
                "plan": f"{investor_type}:{plan}",
                "terms_accepted": "true",
                "terms_accepted_at": terms_at,
                "terms_accepted_ip": terms_ip,
                "terms_version": terms_version
            },
            success_url=url_for("index", _external=True) + "?checkout=success",
            cancel_url=url_for("signup", _external=True) + "?checkout=canceled",
            automatic_tax={"enabled": True},
            allow_promotion_codes=True
        )
        return jsonify({"url": sess.url})
    except Exception as e:
        return (str(e), 500)

# ---------- Stripe Webhook: activate access + send setup email (+ optional SMS) ----------
@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    if not stripe:
        return ("Stripe SDK not installed. Run: pip install stripe", 500)
    if not STRIPE_WEBHOOK_SECRET:
        return ("STRIPE_WEBHOOK_SECRET not set", 500)

    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return (f"Webhook error: {e}", 400)

    etype = event.get("type", "")
    obj = event.get("data", {}).get("object", {}) or {}

    # We activate on Checkout completion OR subscription created OR invoice paid.
    activate_types = {"checkout.session.completed", "customer.subscription.created", "invoice.payment_succeeded"}

    if etype in activate_types:
        # Extract email
        email = ""
        if isinstance(obj, dict):
            email = (obj.get("customer_details") or {}).get("email") or obj.get("customer_email") or ""
            if not email:
                email = (obj.get("metadata") or {}).get("alerts_email") or ""
        email = (email or "").strip().lower()

        # Extract plan/phone
        md = (obj.get("metadata") or {}) if isinstance(obj, dict) else {}
        phone = (md.get("alerts_phone") or "").strip()
        plan  = (md.get("plan") or "").strip()
        customer_id = obj.get("customer") if isinstance(obj, dict) else None

        if email:
            try:
                upsert_user(email=email, phone=phone, plan=plan or None, stripe_customer_id=customer_id, status="active")
            except Exception:
                pass

            # Send setup email only if password not yet set
            try:
                user = get_user_by_email(email)
                has_pw = bool(user and user.get("password_hash"))
                if not has_pw:
                    token = secrets.token_urlsafe(32)
                    expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()
                    mark_setup_token(email, token, expires)
                    setup_link = _external_url(f"/set-password?token={token}")
                    if not BASE_URL:
                        # build with url_for if possible
                        try:
                            setup_link = url_for("set_password", token=token, _external=True)
                        except Exception:
                            pass

                    subj, body = build_setup_email(email, setup_link)
                    send_email(email, subj, body)
            except Exception:
                pass

            # Optional welcome SMS
            if phone:
                send_sms(phone, "Welcome to Signal Hawk Capital! Your subscription is active. Check your email to set up your account password and access your dashboard.")

    return ("ok", 200)

# ---------- Resend setup email (logged out + logged in) ----------
@app.route("/resend-setup", methods=["POST"])
def resend_setup():
    data = request.get_json(force=True) if request.data else {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return ("Email required", 400)

    user = get_user_by_email(email)
    if not user:
        return ("No user found. Please subscribe first.", 404)

    # only resend for active subscribers
    if (user.get("subscription_status") or "").lower() != "active":
        return ("No active subscription. Please subscribe first.", 400)

    # if already has password, direct them to login
    if user.get("password_hash"):
        return jsonify({"ok": True, "message": "Account already set up. Please log in.", "login_url": url_for("login", _external=True)})

    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    mark_setup_token(email, token, expires)

    setup_link = _external_url(f"/set-password?token={token}")
    if not BASE_URL:
        setup_link = url_for("set_password", token=token, _external=True)

    subj, body = build_setup_email(email, setup_link)
    ok, msg = send_email(email, subj, body)
    if not ok:
        return (f"Email send failed: {msg}", 500)

    return jsonify({"ok": True, "message": "Setup email sent."})

# ---------- Password setup (from emailed token) ----------
@app.route("/set-password", methods=["GET", "POST"])
def set_password():
    token = request.args.get("token", "").strip()
    if not token:
        return ("Missing token", 400)

    user = get_user_by_setup_token(token)
    if not user:
        return ("Invalid token", 400)

    exp = (user.get("setup_token_expires") or "").strip()
    if exp:
        try:
            if datetime.fromisoformat(exp) < datetime.utcnow():
                return ("Token expired. Request a new setup email.", 400)
        except Exception:
            pass

    if request.method == "GET":
        return render_template("set_password.html", email=user["email"], token=token)

    pw1 = (request.form.get("password") or "").strip()
    pw2 = (request.form.get("password_confirm") or "").strip()
    if len(pw1) < 10:
        return render_template("set_password.html", email=user["email"], token=token, error="Password must be at least 10 characters.")
    if pw1 != pw2:
        return render_template("set_password.html", email=user["email"], token=token, error="Passwords do not match.")

    set_password(user["email"], pw1)
    clear_setup_token(user["email"])
    login_user(user["email"])
    return redirect(url_for("account"))

# ---------- Password login ----------
@app.route("/login-password", methods=["GET", "POST"])
def login_password():
    if request.method == "GET":
        return render_template("login_password.html")
    email = (request.form.get("email") or "").strip().lower()
    password = (request.form.get("password") or "").strip()
    user = get_user_by_email(email)
    if not user or not user.get("password_hash"):
        return render_template("login_password.html", error="Invalid email or password.")
    if not check_password_hash(user["password_hash"], password):
        return render_template("login_password.html", error="Invalid email or password.")
    login_user(email)
    return redirect(url_for("account"))

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

# ---------- Reset password (request email + token) ----------
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_request():
    if request.method == "GET":
        return render_template("reset_password_request.html")

    email = (request.form.get("email") or "").strip().lower()
    user = get_user_by_email(email)
    # Always respond success (avoid account enumeration)
    if user and (user.get("subscription_status") or "").lower() == "active":
        token = secrets.token_urlsafe(32)
        expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()
        mark_reset_token(email, token, expires)
        reset_link = _external_url(f"/reset-password/confirm?token={token}")
        if not BASE_URL:
            reset_link = url_for("reset_password_confirm", token=token, _external=True)
        subj, body = build_reset_email(email, reset_link)
        send_email(email, subj, body)

    return render_template("reset_password_request.html", success="If an account exists for that email, a reset link has been sent.")

@app.route("/reset-password/confirm", methods=["GET", "POST"])
def reset_password_confirm():
    token = request.args.get("token", "").strip()
    if not token:
        return ("Missing token", 400)
    user = get_user_by_reset_token(token)
    if not user:
        return ("Invalid token", 400)
    exp = (user.get("reset_token_expires") or "").strip()
    if exp:
        try:
            if datetime.fromisoformat(exp) < datetime.utcnow():
                return ("Token expired. Request a new reset email.", 400)
        except Exception:
            pass

    if request.method == "GET":
        return render_template("reset_password_confirm.html", email=user["email"], token=token)

    pw1 = (request.form.get("password") or "").strip()
    pw2 = (request.form.get("password_confirm") or "").strip()
    if len(pw1) < 10:
        return render_template("reset_password_confirm.html", email=user["email"], token=token, error="Password must be at least 10 characters.")
    if pw1 != pw2:
        return render_template("reset_password_confirm.html", email=user["email"], token=token, error="Passwords do not match.")

    set_password(user["email"], pw1)
    clear_reset_token(user["email"])
    login_user(user["email"])
    return redirect(url_for("account"))

# ---------- Billing portal (cancel, change plan, invoices) ----------
@app.route("/billing-portal", methods=["POST"])
@login_required
def billing_portal():
    if not stripe:
        return ("Stripe SDK not installed. Run: pip install stripe", 500)
    user = get_current_user()
    if not user or not user.get("stripe_customer_id"):
        return ("No Stripe customer on file", 400)

    portal = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=url_for("account", _external=True),
    )
    return jsonify({"url": portal.url})

# ---------- Admin: list users (protect in production) ----------
@app.route("/admin/users")
def admin_users():
    try:
        with closing(db_connect()) as conn:
            rows = conn.execute("""
                SELECT email, phone, plan, stripe_customer_id, subscription_status,
                       terms_accepted_at, terms_accepted_ip, terms_version,
                       created_at, updated_at
                FROM users
                ORDER BY updated_at DESC
            """).fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return (str(e), 500)

# ---------- Account page ----------
@app.route("/account")
def account():
    user = get_current_user()
    # Render your existing account template if present; fallback to a minimal view.
    try:
        return render_template("account.html", user=user)
    except Exception:
        # Minimal fallback
        if not user:
            return redirect(url_for("login"))
        return jsonify(dict(user))

# ---------- Protect paid routes ----------
# /signals should require active subscription
try:
    # Replace existing signals view function by wrapping it if it already exists
    _old_signals_view = app.view_functions.get("signals")
    if _old_signals_view:
        app.view_functions["signals"] = subscription_required(_old_signals_view)
except Exception:
    pass

try:
    _old_historical_view = app.view_functions.get("historical")
    if _old_historical_view:
        app.view_functions["historical"] = subscription_required(_old_historical_view)
except Exception:
    pass

# ---------- Static PDFs (update filenames to match your actual names) ----------
@app.route("/whitepaper")
def whitepaper():
    return send_file("static/Signal_Hawk_Whitepaper.pdf", as_attachment=True)

# If you use HTML pages for terms/disclosures/privacy, keep these routes as templates instead.
# Otherwise, serve PDFs directly:
@app.route("/terms-pdf")
def terms_pdf():
    return send_file("static/Signal_Hawk_Terms.pdf", as_attachment=False)

@app.route("/disclosures-pdf")
def disclosures_pdf():
    return send_file("static/Signal_Hawk_Disclosures.pdf", as_attachment=False)

@app.route("/privacy-pdf")
def privacy_pdf():
    return send_file("static/Signal_Hawk_Privacy.pdf", as_attachment=False)

