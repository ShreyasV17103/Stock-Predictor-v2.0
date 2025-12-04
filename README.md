# 📈 Stock Prediction v2.0

A powerful, all-in-one Python tool that predicts Indian Stock Market (NSE) prices using **8 advanced Machine Learning & Deep Learning algorithms**.

It automatically fetches historical data (from Year 2000), runs multiple models in parallel, compares their accuracy, and generates a detailed CSV report—**all without requiring any API keys.**

---

## ✨ Key Features

* **Zero API Keys Needed:** Uses `nselib` to fetch official NSE data legally and for free.
* **8 Powerful Algorithms:** Runs XGBoost, Random Forest, LightGBM, CatBoost, Gradient Boosting, MLP (Neural Network), GRU, and 1D-CNN.
* **Smart Date Logic:** Automatically detects if the prediction is for "Today" or "Tomorrow" based on market closing times.
* **Dual Trend Analysis:** Calculates both **Net Change** (Overnight hold) and **Intraday Change** (Day trading) to prevent confusion between Red/Green candles and actual trend.
* **Auto-Export:** Saves all predictions to a timestamped CSV file (`STOCKNAME_prediction_report.csv`).
* **Deep Learning Ready:** Includes PyTorch implementations for GRU and CNN models.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.8 or higher.
* An active internet connection (to fetch NSE data).

### 2. Install Dependencies
Open your terminal in the project folder and run:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost nselib pandas_market_calendars torch
