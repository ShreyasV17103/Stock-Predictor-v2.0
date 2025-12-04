import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from nselib import capital_market

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

import torch
import torch.nn as nn

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
    except Exception:
        return user_input.upper()

def get_data(symbol):
    print(f"   -> Fetching MAXIMUM history for {symbol} (from Year 2000)...")
    try:
        to_date = datetime.now().strftime("%d-%m-%Y")
        data = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol,
            from_date='01-01-2000',
            to_date=to_date
        )

        if data is None or data.empty:
            print("      Warning: Long history failed. Retrying with last 10 years...")
            data = capital_market.price_volume_and_deliverable_position_data(
                symbol=symbol,
                period='10Y'
            )

        if data is None or data.empty:
            return None

        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')
        data = data.set_index('Date').sort_index()
        data.rename(
            columns={
                'OpenPrice': 'Open',
                'HighPrice': 'High',
                'LowPrice': 'Low',
                'ClosePrice': 'Close',
                'TotalTradedQuantity': 'Volume'
            },
            inplace=True
        )

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
    except Exception:
        return 0.0

def build_features(df):
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['Target_Close'] = df['Close'].shift(-1)
    df['Target_Open'] = df['Open'].shift(-1)
    df.dropna(inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close', 'MA5']
    X = df[features]
    y_close = df['Target_Close']
    y_open = df['Target_Open']
    weights = np.linspace(0.01, 1.0, len(df))
    return df, X, y_close, y_open, features, weights

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

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
    train_data, test_data = dataset[0:training_size, :], dataset[training_size:len(dataset), :]

    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)

    X_train_torch = torch.from_numpy(X_train).type(torch.Tensor).unsqueeze(2)
    y_train_torch = torch.from_numpy(y_train).type(torch.Tensor).unsqueeze(1)
    X_test_torch = torch.from_numpy(X_test).type(torch.Tensor).unsqueeze(2)

    model = CNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(EPOCHS):
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

    score = r2_score(y_test_inv[0], test_predict_inv[:, 0]) * 100

    last_chunk = dataset[len(dataset) - LOOK_BACK:]
    last_chunk_torch = torch.from_numpy(last_chunk).type(torch.Tensor).unsqueeze(0)

    with torch.no_grad():
        future_pred_scaled = model(last_chunk_torch).numpy()

    future_pred = scaler.inverse_transform(future_pred_scaled)
    return future_pred[0][0], score

def train_pytorch_gru(data_series):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data_series.values.reshape(-1, 1))

    training_size = int(len(dataset) * 0.85)
    train_data, test_data = dataset[0:training_size, :], dataset[training_size:len(dataset), :]

    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)

    X_train_torch = torch.from_numpy(X_train).type(torch.Tensor).unsqueeze(2)
    y_train_torch = torch.from_numpy(y_train).type(torch.Tensor).unsqueeze(1)
    X_test_torch = torch.from_numpy(X_test).type(torch.Tensor).unsqueeze(2)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    model = GRUModel(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(EPOCHS):
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

    score = r2_score(y_test_inv[0], test_predict_inv[:, 0]) * 100

    last_chunk = dataset[len(dataset) - LOOK_BACK:]
    last_chunk_torch = torch.from_numpy(last_chunk).type(torch.Tensor).unsqueeze(0)

    with torch.no_grad():
        future_pred_scaled = model(last_chunk_torch).numpy()

    future_pred = scaler.inverse_transform(future_pred_scaled)
    return future_pred[0][0], score

def run_catboost(df_fe, X, y_close, y_open, features, weights, ticker):
    if len(df_fe) < 5:
        print("Error: Not enough continuous data for CatBoost.")
        return
    print("   -> Training CatBoost Models...")
    model_close = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df_fe.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----CatBoost Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_lightgbm(df_fe, X, y_close, y_open, features, weights, ticker):
    if len(df_fe) < 5:
        print("Error: Not enough continuous data for LightGBM.")
        return
    print("   -> Training LightGBM Models...")
    model_close = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df_fe.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----LightGBM Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_xgboost(df_fe, X, y_close, y_open, features, weights, ticker):
    if len(df_fe) < 5:
        print("Error: Not enough continuous data for XGBoost.")
        return
    print("   -> Training XGBoost Models...")
    model_close = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df_fe.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----XGBoost Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_gradient_boosting(df_fe, X, y_close, y_open, features, weights, ticker):
    if len(df_fe) < 5:
        print("Error: Not enough continuous data for Gradient Boosting.")
        return
    print("   -> Training Gradient Boosting Models...")
    model_close = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df_fe.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----Gradient Boosting Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_random_forest(df_fe, X, y_close, y_open, features, weights, ticker):
    if len(df_fe) < 5:
        print("Error: Not enough continuous data for Random Forest.")
        return
    print("   -> Training Random Forest Models...")
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X, y_close, sample_weight=weights)

    model_open = RandomForestRegressor(n_estimators=100, random_state=42)
    model_open.fit(X, y_open, sample_weight=weights)

    score_close = r2_score(y_close, model_close.predict(X)) * 100
    score_open = r2_score(y_open, model_open.predict(X)) * 100

    last_row = df_fe.iloc[[-1]][features]
    pred_close = model_close.predict(last_row)[0]
    pred_open = model_open.predict(last_row)[0]

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----Random Forest Prediction (Weighted History)------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_mlp(df_fe, X, y_close, y_open, features, ticker):
    if len(df_fe) < 20:
        print("Error: Not enough continuous data for MLP.")
        return
    print("--- Training Neural Network (MLP) ---")

    scaler_X = MinMaxScaler()
    scaler_y_close = MinMaxScaler()
    scaler_y_open = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_close_scaled = scaler_y_close.fit_transform(y_close.values.reshape(-1, 1))
    y_open_scaled = scaler_y_open.fit_transform(y_open.values.reshape(-1, 1))

    model_close = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model_close.fit(X_scaled, y_close_scaled.ravel())

    model_open = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model_open.fit(X_scaled, y_open_scaled.ravel())

    last_row_scaled = scaler_X.transform(df_fe.iloc[[-1]][features])

    pred_close_scaled = model_close.predict(last_row_scaled)
    pred_open_scaled = model_open.predict(last_row_scaled)

    pred_close = scaler_y_close.inverse_transform(pred_close_scaled.reshape(-1, 1))[0][0]
    pred_open = scaler_y_open.inverse_transform(pred_open_scaled.reshape(-1, 1))[0][0]

    preds_close_all = scaler_y_close.inverse_transform(model_close.predict(X_scaled).reshape(-1, 1))
    real_close_all = scaler_y_close.inverse_transform(y_close_scaled)
    score_close = r2_score(real_close_all, preds_close_all) * 100

    preds_open_all = scaler_y_open.inverse_transform(model_open.predict(X_scaled).reshape(-1, 1))
    real_open_all = scaler_y_open.inverse_transform(y_open_scaled)
    score_open = r2_score(real_open_all, preds_open_all) * 100

    current_close = df_fe.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----Neural Network (MLP) Prediction------")
    print(f"Model Accuracy (Close Price): {score_close:.2f}%")
    print(f"Model Accuracy (Open Price):  {score_open:.2f}%")
    print("-" * 30)
    print(f"Yesterday Closing Price for {ticker}: {current_close:.2f}")
    print(f"Yesterday Opening Price for {ticker}: {df_fe.iloc[-1]['Open']:.2f}")
    print()
    print(f"Predicted Closing Price for {ticker}: {pred_close:.2f}")
    print(f"Predicted Opening Price for {ticker}: {pred_open:.2f}")
    print(f"Predicted Day Change: {pct_change:.2f}%")
    if pct_change > 0:
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_cnn(df, ticker):
    df = df.copy().dropna()
    if df.shape[0] < 50:
        print("Error: Not enough continuous data for 1D CNN (Need > 50 days).")
        return
    print("--- Training PyTorch 1D-CNN Models ---")
    print("1. Training Closing Price Model...")
    pred_close, score_close = train_pytorch_cnn(df['Close'])
    print("2. Training Opening Price Model...")
    pred_open, score_open = train_pytorch_cnn(df['Open'])

    current_close = df.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----PyTorch 1D-CNN Prediction Result------")
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
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def run_gru(df, ticker):
    df = df.copy().dropna()
    if df.shape[0] < 50:
        print("Error: Not enough continuous data for GRU (Need > 50 days).")
        return
    print("--- Training PyTorch GRU Models ---")
    print("1. Training Closing Price Model...")
    pred_close, score_close = train_pytorch_gru(df['Close'])
    print("2. Training Opening Price Model...")
    pred_open, score_open = train_pytorch_gru(df['Open'])

    current_close = df.iloc[-1]['Close']
    pct_change = day_change(pred_close, current_close)

    print("\n-----PyTorch GRU Prediction Result------")
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
        print("Predicted Trend: Bullish 📈")
    elif pct_change < 0:
        print("Predicted Trend: Bearish 📉")
    else:
        print("Predicted Trend: Neutral 📊")

def main():
    user_input = input("Enter the stock name: ").strip()
    ticker = get_ticker_from_name(user_input)

    df = get_data(ticker)
    if df is None or df.empty:
        print(f"Error: No data found for {ticker}.")
        return

    df = df.dropna()
    if df.shape[0] < 5:
        print(f"Error: Not enough continuous data to train for {ticker}.")
        return

    df_fe, X, y_close, y_open, features, weights = build_features(df)

    print("\nRunning ALL models...\n")

    run_catboost(df_fe, X, y_close, y_open, features, weights, ticker)
    run_lightgbm(df_fe, X, y_close, y_open, features, weights, ticker)
    run_xgboost(df_fe, X, y_close, y_open, features, weights, ticker)
    run_gradient_boosting(df_fe, X, y_close, y_open, features, weights, ticker)
    run_random_forest(df_fe, X, y_close, y_open, features, weights, ticker)
    run_mlp(df_fe, X, y_close, y_open, features, ticker)
    run_cnn(df, ticker)
    run_gru(df, ticker)

if __name__ == "__main__":
    main()
