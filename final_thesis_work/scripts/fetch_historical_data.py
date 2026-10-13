"""
fetch_historical_data.py
Downloads S&P 500 logarithmic daily returns for 4 historical epochs and caches
them as NumPy .npy files in the data/ directory for use by the Flask API.

The arrays are not stored in the repository (the data come from Yahoo Finance via yfinance).
After downloading, each array is checked against data/historical/checksums.json (element count and
SHA-256 of the float64 values) recorded from the arrays used for the thesis dashboard. Yahoo can revise
its history, so a mismatch is reported (exit code 2) rather than silently accepted.

    python3 scripts/fetch_historical_data.py [--out-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: python3 -m pip install yfinance")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'historical')
CHECKSUMS = os.path.join(DATA_DIR, 'checksums.json')

# Define the historical scenarios with their exact date windows
HISTORICAL_SCENARIOS = {
    "real_2000_dotcom": {
        "label":       "2000 Dot-Com Bubble Collapse",
        "ticker":      "^GSPC",
        "start":       "2000-01-01",
        "end":         "2002-10-31",
        "description": "The catastrophic Nasdaq-led unwinding of overvalued tech equities. A prolonged, multi-year slow regime decay — validating the HMM's ability to detect sustained bear regimes."
    },
    "real_2008_crisis": {
        "label":       "2008 Global Financial Crisis",
        "ticker":      "^GSPC",
        "start":       "2007-06-01",
        "end":         "2009-06-30",
        "description": "The systemic collapse of the U.S. housing market propagating into global credit freezes. Extreme Student-t fat tails with sudden high-volatility regime transitions."
    },
    "real_2020_covid": {
        "label":       "2020 COVID-19 Crash & Recovery",
        "ticker":      "^GSPC",
        "start":       "2020-01-01",
        "end":         "2020-12-31",
        "description": "A historically unprecedented V-shaped crash and recovery (33% drawdown in 33 days). Tests CPPI survival floors and HMM detection speed under extreme non-stationarity."
    },
    "real_2013_bull": {
        "label":       "2013–2019 Extended Bull Market",
        "ticker":      "^GSPC",
        "start":       "2013-01-01",
        "end":         "2019-12-31",
        "description": "A sustained low-volatility stationary regime. Analogous to the Flat MAB in theory — validates that Kelly-based approaches exploit stable environments as predicted."
    },
}

def fetch_and_save(key, config):
    print(f"  Fetching: {config['label']} ({config['start']} → {config['end']})...")
    ticker = yf.Ticker(config["ticker"])
    hist = ticker.history(start=config["start"], end=config["end"])

    if hist.empty:
        print(f"  WARNING: No data returned for {key}. Skipping.")
        return False

    prices = hist["Close"].dropna().values
    if len(prices) < 5:
        print(f"  WARNING: Not enough data for {key}. Skipping.")
        return False

    # Compute log-returns (the standard tool for our WealthTracker)
    log_returns = np.diff(np.log(prices))

    out_path = os.path.join(OUT_DIR, f"{key}.npy")
    np.save(out_path, log_returns)
    print(f"  ✅ Saved {len(log_returns)} daily log-returns → {out_path}")
    return True

def verify(key):
    """Compare a saved array with checksums.json. Returns True if it matches."""
    expected = json.load(open(CHECKSUMS)).get(key)
    if expected is None:
        print(f"  {key}: no recorded checksum")
        return False
    a = np.load(os.path.join(OUT_DIR, f"{key}.npy"))
    got = hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()
    ok = a.size == expected["n"] and got == expected["sha256_float64_le"]
    print(f"  {key}: {'OK' if ok else 'MISMATCH'} (n={a.size}, expected {expected['n']})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DATA_DIR)
    OUT_DIR = ap.parse_args().out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n=== Fetching Historical S&P 500 Data ===\n")
    success_count = 0
    fetched = []
    for key, config in HISTORICAL_SCENARIOS.items():
        ok = fetch_and_save(key, config)
        if ok:
            success_count += 1
            fetched.append(key)

    print(f"\n✅ Done. {success_count}/{len(HISTORICAL_SCENARIOS)} scenarios cached successfully.")
    print(f"   Data directory: {OUT_DIR}")
    print("\n=== Checking against data/historical/checksums.json ===\n")
    matches = [verify(k) for k in fetched]
    if success_count < len(HISTORICAL_SCENARIOS) or not all(matches):
        print("\nWARNING: the downloaded data differ from the arrays the dashboard results were computed on.")
        sys.exit(2)
