"""Live data pull for main.ipynb.

Rates: FRED Treasury constant-maturity par yields (proxy for SOFR OIS; replace with Bloomberg SOFR OIS quotes if available).
Options: SPX chain from yfinance, cleaned exactly as in the notebook (OTM only, bid > 0.05, intrinsic and monotonicity checks,
puts converted to call-equivalents with the same curve). Writes calibration_data_<Month>_<DD>_<YYYY>.xlsx (sheets: Cleaned Data,
Original Data, Spot Price, Implied Q, Rates) and rates_<Month>_<DD>_<YYYY>.csv next to this file. Then set DATA_FILE in main.ipynb.

Run during liquid hours (SPX options quote until 16:15 New York; the index stops at 16:00).
"""
import os, io, sys, urllib.request, json
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
import heston_pricer
from heston_pricer import DF

# ---------------- 1. Rates: Treasury constant-maturity par yields from FRED ----------------
series = {"DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12, "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0,
          "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0}
rows = []
for s, T in series.items():
    raw = urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw)); df = df[df[s] != '.']
    rows.append({"tenor_years": T, "rate_pct": float(df.iloc[-1][s]), "as_of": df.iloc[-1]["observation_date"], "series": s,
                 "source": "FRED Treasury constant-maturity yield (proxy for SOFR OIS)"})
rates = pd.DataFrame(rows).sort_values("tenor_years")
today = datetime.now()
rates_file = f"rates_{today.strftime('%B_%d_%Y')}.csv"
rates.to_csv(rates_file, index=False)
print(rates.to_string()); print("saved", rates_file)


# Treasury par yields: semi-annual coupons from 2y. The notebook rebuilds the same curve from the "Rates" sheet.
heston_pricer.set_rate_curve(list(zip(rates["tenor_years"], rates["rate_pct"])), coupon_freq=2)
print("zero rates check:", {t: round(float(heston_pricer.rate_curve_func(t)) * 100, 3) for t in [0.25, 1, 2, 4]})


# ---------------- 2. Options: the notebook's get_clean_calibration_data with a wider expiry window ----------------
def get_clean_calibration_data(ticker_symbol, min_volume=1, max_days=1600):
    print(f"Retrieving data for {ticker_symbol}")
    asset = yf.Ticker(ticker_symbol)
    history = asset.history(period="1d")
    S0 = history['Close'].iloc[-1]
    print(f"Spot Price (S0): {S0:.2f}")
    expirations = asset.options
    today = datetime.now()
    selected_dates = [exp for exp in expirations if 30 < (datetime.strptime(exp, "%Y-%m-%d") - today).days < max_days]
    all_dfs, raw_dfs, q_rows = [], [], []
    for exp in selected_dates:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        T = (exp_date - today).days / 365.0
        df_r = DF(T)
        try:
            chain = asset.option_chain(exp)
            calls = chain.calls.copy(); puts = chain.puts.copy()
        except Exception as e:
            print("skip", exp, e); continue
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
        try:
            common_strikes = np.intersect1d(calls['strike'].values, puts['strike'].values)
            if len(common_strikes) == 0: continue
            K_atm = common_strikes[np.abs(common_strikes - S0).argmin()]
            C_atm = calls[calls['strike'] == K_atm]['mid_price'].values[0]
            P_atm = puts[puts['strike'] == K_atm]['mid_price'].values[0]
            parity_val = (C_atm - P_atm + K_atm * df_r) / S0
            implied_q = 0 if parity_val <= 0 else -np.log(parity_val) / T
            df_q = np.exp(-implied_q * T)
            q_rows.append({'expiry': exp, 'T': T, 'K_atm': K_atm, 'C_atm': C_atm, 'P_atm': P_atm, 'implied_q': implied_q})
        except Exception as e:
            print(f"Skipping exp {exp}: Error calc q ({e})"); continue
        puts_otm = puts[(puts['strike'] < S0) & (puts['strike'] > S0 * 0.45) & (puts['volume'] >= min_volume)].copy(); puts_otm['type'] = 'put'
        calls_otm = calls[(calls['strike'] >= S0) & (calls['strike'] < S0 * 1.25) & (calls['volume'] >= min_volume)].copy(); calls_otm['type'] = 'call'
        df = pd.concat([puts_otm, calls_otm])
        if df.empty: continue
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['T'] = T; df['q'] = implied_q
        raw_dfs.append(df.copy())
        liq_mask = df['bid'] > 0.05
        n_liq = int((~liq_mask).sum()); df = df[liq_mask].copy()
        S_disc = S0 * df_q; K_disc = df['strike'] * df_r
        intrinsic_val = np.where(df['type'] == 'call', np.maximum(0, S_disc - K_disc), np.maximum(0, K_disc - S_disc))
        valid = df['mid_price'] > (intrinsic_val + 0.01)
        n_arb = int((~valid).sum()); df = df[valid].copy()
        if df.empty: continue
        df['market_price_call_equivalent'] = np.where(df['type'] == 'call', df['mid_price'], df['mid_price'] + S_disc - df['strike'] * df_r)
        df = df.sort_values('strike').reset_index(drop=True)
        rolling_min = df['market_price_call_equivalent'].cummin()
        mono = df['market_price_call_equivalent'] <= (rolling_min + 0.10)
        n_mono = int((~mono).sum()); df = df[mono].copy()
        print(f"  {exp} T={T:.3f} q={implied_q:+.4f}: kept {len(df)} (dropped liq {n_liq}, arb {n_arb}, mono {n_mono})")
        clean_df = df[['strike', 'T', 'market_price_call_equivalent', 'impliedVolatility', 'type', 'mid_price', 'q']]
        clean_df = clean_df.rename(columns={'type': 'type_before_conversion', 'impliedVolatility': 'implied_vol', 'q': 'implied_q'})
        all_dfs.append(clean_df)
    final_df = pd.concat(all_dfs, ignore_index=True).sort_values(['T', 'strike']).reset_index(drop=True)
    raw_df = pd.concat(raw_dfs, ignore_index=True)
    s0_df = pd.DataFrame([{'S0': S0, 'Ticker': ticker_symbol, 'Date': today}])
    for col in raw_df.columns:
        if pd.api.types.is_datetime64_any_dtype(raw_df[col]) and raw_df[col].dt.tz is not None:
            raw_df[col] = raw_df[col].dt.tz_localize(None)
    filename = f"calibration_data_{today.strftime('%B_%d_%Y')}.xlsx"
    with pd.ExcelWriter(filename) as writer:
        final_df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        raw_df.to_excel(writer, sheet_name="Original Data", index=False)
        s0_df.to_excel(writer, sheet_name="Spot Price", index=False)
        pd.DataFrame(q_rows).to_excel(writer, sheet_name="Implied Q", index=False)
        rates.to_excel(writer, sheet_name="Rates", index=False)
    print(f"Data saved to {filename}; total options: {len(final_df)}; snapshot {today}")
    return S0, final_df

S0, data = get_clean_calibration_data("^SPX", max_days=1600)
print(data.groupby('T').size().to_string())
