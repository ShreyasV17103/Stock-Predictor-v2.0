import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from nselib import capital_market
from datetime import datetime

# --- CONFIG ---
LOOK_BACK = 10
EPOCHS = 100

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

# --- PyTorch CNN Model Definition ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Conv1D: Scans pattern. in_channels=1 (price), out_channels=64 (filters)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # Calculate size after Conv + Pool (Logic helper)
        # Input length (Lookback) = 10
        # After Conv (k=2): 9
        # After Pool (k=2): 4
        self.fc1 = nn.Linear(64 * 4, 50) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: [batch, lookback, 1] -> needs [batch, channels, length] for Conv1D
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_pytorch_cnn(data_series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data_series.values.reshape(-1, 1))
    
    training_size = int(len(dataset) * 0.85)
    train_data, test_data = dataset[0:training_size,:], dataset[training_size:len(dataset),:]
    
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)
    
    X_train_torch = torch.from_numpy(X_train).type(torch.Tensor).unsqueeze(2)
    y_train_torch = torch.from_numpy(y_train).type(torch.Tensor).unsqueeze(1)
    X_test_torch = torch.from_numpy(X_test).type(torch.Tensor).unsqueeze(2)
    
    model = CNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for t in range(EPOCHS):
        y_train_pred = model(X_train_torch)
        loss = criterion(y_train_pred, y_train_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_predict = model(X_test_torch).numpy()
        
    y_test_inv = scaler.inverse_transform([y_test])
    test_predict_inv = scaler.inverse_transform(test_predict)
    
    score = r2_score(y_test_inv[0], test_predict_inv[:,0]) * 100
    
    # --- FIX: Removed extra unsqueeze ---
    last_chunk = dataset[len(dataset)-LOOK_BACK:]
    last_chunk_torch = torch.from_numpy(last_chunk).type(torch.Tensor).unsqueeze(0)
    
    with torch.no_grad():
        future_pred_scaled = model(last_chunk_torch).numpy()
        
    future_pred = scaler.inverse_transform(future_pred_scaled)
    return future_pred[0][0], score

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

    df.dropna(inplace=True)
    if df.shape[0] < 50:
        print(f"Error: Not enough continuous data (Need > 50 days).")
        return

    print("--- Training PyTorch CNN Models ---")
    
    print("1. Training Closing Price Model...")
    pred_close, score_close = train_pytorch_cnn(df['Close'])
    
    print("2. Training Opening Price Model...")
    pred_open, score_open = train_pytorch_cnn(df['Open'])

    current_close = df.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print(f"\n-----PyTorch 1D-CNN Prediction Result------")
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