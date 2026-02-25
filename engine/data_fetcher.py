"""
Data Fetcher — Real SPX + VIX Historical Data
================================================
Downloads daily OHLC from Yahoo Finance for ^GSPC (SPX) and ^VIX,
merges them, and caches to CSV for backtesting.

Run standalone:   python -m engine.data_fetcher
Or imported:      from engine.data_fetcher import fetch_and_cache, load_cached_data
"""

import os
from datetime import date, timedelta
from typing import Optional, List

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "spx_vix_daily.csv")


def fetch_and_cache(
    start: str = "2020-01-02",
    end: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download SPX + VIX daily data from Yahoo Finance and save to CSV.
    
    Returns path to the cached CSV file.
    Requires: pip install yfinance pandas
    """
    if os.path.exists(CACHE_FILE) and not force:
        # Check if cache is recent (within 1 day)
        import time
        age_hours = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600
        if age_hours < 18:
            print(f"Using cached data: {CACHE_FILE} ({age_hours:.1f}h old)")
            return CACHE_FILE

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "yfinance and pandas required for real data. "
            "Install with: pip install yfinance pandas"
        )

    if end is None:
        end = date.today().isoformat()

    print(f"Downloading SPX data ({start} to {end})...")
    spx = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    
    print(f"Downloading VIX data ({start} to {end})...")
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)

    if spx.empty or vix.empty:
        raise RuntimeError("Failed to download data from Yahoo Finance")

    # Handle multi-level columns from yfinance
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Rename columns
    spx = spx.rename(columns={
        "Open": "spx_open", "High": "spx_high",
        "Low": "spx_low", "Close": "spx_close",
    })[["spx_open", "spx_high", "spx_low", "spx_close"]]

    vix = vix.rename(columns={
        "Open": "vix_open", "Close": "vix_close",
    })[["vix_open", "vix_close"]]

    # Merge on date
    merged = spx.join(vix, how="inner")
    merged.index.name = "date"

    # Drop any rows with NaN
    merged = merged.dropna()

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(CACHE_FILE)
    print(f"Saved {len(merged)} trading days to {CACHE_FILE}")
    return CACHE_FILE


def load_cached_data() -> Optional[str]:
    """Return path to cached CSV if it exists, else None."""
    if os.path.exists(CACHE_FILE):
        return CACHE_FILE
    return None


def get_data_info() -> dict:
    """Return info about cached data for the UI."""
    if not os.path.exists(CACHE_FILE):
        return {"available": False, "path": None, "rows": 0, "date_range": None}
    
    try:
        with open(CACHE_FILE, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return {"available": False, "path": CACHE_FILE, "rows": 0, "date_range": None}
        
        first_date = lines[1].split(",")[0]
        last_date = lines[-1].split(",")[0]
        return {
            "available": True,
            "path": CACHE_FILE,
            "rows": len(lines) - 1,
            "date_range": f"{first_date} to {last_date}",
        }
    except Exception:
        return {"available": False, "path": CACHE_FILE, "rows": 0, "date_range": None}


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download SPX + VIX data")
    parser.add_argument("--start", default="2020-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    path = fetch_and_cache(args.start, args.end, args.force)
    
    # Quick summary
    info = get_data_info()
    print(f"\nData ready: {info['rows']} days, {info['date_range']}")
