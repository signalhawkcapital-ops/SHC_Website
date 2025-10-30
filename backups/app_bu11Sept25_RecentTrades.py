import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for
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

def download_data(tickers, start, end, retries=3, delay=5):
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, interval='1d', auto_adjust=False)['Close']
            return data
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise Exception("Failed to download data after retries")

def simulate_spxl_data(gspc_data, start_date, spxl_inception_date=None, spxl_inception_price=None):
    expense_ratio = 0.0787  # 0.87% annual expense ratio
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
    data['SMA_80'] = data[spxl_column].rolling(window=57).mean()
    data['SMA_80_Slope'] = data['SMA_80'].diff()
    
    strategy_value = 1.0
    strategy_returns = []
    holding = True
    prev_price = data[spxl_column].iloc[0]
    buy_signals = []
    sell_signals = []
    expense_ratio = 0.0887  # 0.87% annual expense ratio
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
                daily_return -= daily_expense
            strategy_value *= daily_return if daily_return > 0 else 1
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
                daily_return -= daily_expense
            strategy_value *= daily_return if daily_return > 0 else 1
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

        # Download data
        raw_data = download_data(all_tickers, start_date, end_date)
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

        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if data.empty:
            raise ValueError("No data available for the selected date range.")
        
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
            recent_trades_html = build_recent_trades_html_from_data(data, use_simulated=False, limit=30)
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

        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        if data.empty:
            raise ValueError("No data available for the selected date range.")
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

if __name__ == "__main__":
    print("App initialized")
    app.run(debug=True)