import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
from sklearn.metrics import r2_score
from nselib import capital_market

def get_ticker_from_name(user_input):
    print(f"Searching for {user_input}...")
    try:
        df = capital_market.equity_list()
        df.columns = [c.strip().upper() for c in df.columns]
        mask = df['NAME OF COMPANY'].str.contains(user_input, case=False, na=False)
        results = df[mask]
        
        if results.empty:
            clean_names = df['NAME OF COMPANY'].str.replace(' ', '')
            clean_input = user_input.replace(' ', '')
            mask = clean_names.str.contains(clean_input, case=False, na=False)
            results = df[mask]

        if not results.empty:
            found_sym = results.iloc[0]['SYMBOL']
            print(f"   -> Found: {results.iloc[0]['NAME OF COMPANY']} ({found_sym})")
            return found_sym
        else:
            return user_input.upper()
    except:
        return user_input.upper()

def get_data(symbol):
    print(f"   -> Fetching MAXIMUM history for {symbol} (from Year 2000)...")
    try:
        to_date = datetime.now().strftime("%d-%m-%Y")
        data = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, from_date='01-01-2000', to_date=to_date)
        
        if data is None or data.empty: 
            print("      Warning: Long history failed. Retrying with last 10 years...")
            data = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period='10Y')

        if data is None or data.empty: return None

        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')
        data = data.set_index('Date').sort_index()
        data.rename(columns={'OpenPrice':'Open', 'HighPrice':'High', 'LowPrice':'Low', 'ClosePrice':'Close', 'TotalTradedQuantity':'Volume'}, inplace=True)
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).str.replace(',', '')
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        print(f"      Successfully loaded {len(data)} days of data!")
        return data[cols]
    except Exception as e: 
        print(f"      Error fetching data: {e}")
        return None

def day_change(new_price, old_price):
    try:
        return ((new_price - old_price) / old_price) * 100
    except:
        return 0.0

def main():
    user_input = input("Enter the stock name: ").strip()
    ticker = get_ticker_from_name(user_input)
    
    df = get_data(ticker)
    if df is None or df.empty: 
        print(f"Error: No data found for {ticker}.")
        return

    df['Prev_Close'] = df['Close'].shift(1)
    df['Target_Close'] = df['Close'].shift(-1)
    df['Target_Open'] = df['Open'].shift(-1)
    
    df.dropna(inplace=True)
    if df.shape[0] < 5:
        print(f"Error: Not enough continuous data to train for {ticker}.")
        return

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close']
    X = df[features]
    y_close = df['Target_Close']
    y_open = df['Target_Open']

    # --- RECENCY WEIGHTING ---
    weights = np.linspace(0.01, 1.0, len(df))

    # Train Models
    model_close = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]
    
    current_close = df.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print(f"\n-----Gradient Boosting Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish \U0001F4C8")
    elif pct_change < 0:
        print("Predicted Trend: Bearish \U0001F4C9")
    else:
        print("Predicted Trend: Neutral \U0001F4CA")

if __name__ == "__main__":
    main()