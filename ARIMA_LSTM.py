import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# Load data
arima_data = pd.read_csv("arima_stock_data.csv")
X_lstm = np.load("lstm_X.npy")
Y_lstm = np.load("lstm_Y.npy")

# Extract close prices for comparison
close_prices = arima_data["Close"].values

# --- 1. Train ARIMA Model ---
def train_arima(data, order=(5,1,0)):  # ARIMA(p,d,q)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    model = ARIMA(train, order=order)
    arima_fit = model.fit()

    predictions = arima_fit.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, predictions)
    rmse = math.sqrt(mean_squared_error(test, predictions))
    
    return predictions, test, mae, rmse, arima_fit

arima_pred, arima_actual, arima_mae, arima_rmse, arima_model = train_arima(close_prices)

# --- 2. Train LSTM Model ---
def train_lstm(X, Y, epochs=50, batch_size=16):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    lstm_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(Y_test, lstm_pred)
    rmse = math.sqrt(mean_squared_error(Y_test, lstm_pred))
    
    return lstm_pred, Y_test, mae, rmse, model

X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))  # Reshape for LSTM
lstm_pred, lstm_actual, lstm_mae, lstm_rmse, lstm_model = train_lstm(X_lstm, Y_lstm)

def create_sliding_window(df, target_col, window_size=10):
    X, Y = [], []
    
    for i in range(len(df) - window_size):
        X.append(df[target_col].iloc[i : i + window_size].values)
        Y.append(df[target_col].iloc[i + window_size])
    
    return np.array(X), np.array(Y)


# --- 3. Hybrid Model: ARIMA + LSTM ---
def train_hybrid(arima_model, original_data, X_lstm, Y_lstm, train_size=0.8):
    """
    Trains a hybrid ARIMA-LSTM model where:
    - ARIMA models the trend
    - LSTM models the residuals
    """
    # Step 1: Get ARIMA predicted trend
    train_split = int(len(original_data) * train_size)
    arima_fitted_values = arima_model.fittedvalues  # ARIMA fitted on training data

    # Ensure the residuals are computed correctly
    residuals = original_data[:len(arima_fitted_values)] - arima_fitted_values

    # Step 2: Train LSTM on residuals
    residual_X, residual_Y = create_sliding_window(pd.DataFrame(residuals, columns=["Residual"]), "Residual", window_size=10)

    # Ensure correct reshaping for LSTM
    residual_X = residual_X.reshape((residual_X.shape[0], residual_X.shape[1], 1))

    # Train LSTM
    hybrid_pred, hybrid_actual, hybrid_mae, hybrid_rmse, hybrid_model = train_lstm(residual_X, residual_Y)

    # Step 3: Align sizes correctly
    arima_test_pred = arima_model.forecast(steps=len(hybrid_pred))  # Forecast ARIMA over LSTM test range

    # Ensure we are adding matching elements
    if len(arima_test_pred) != len(hybrid_pred):
        min_len = min(len(arima_test_pred), len(hybrid_pred))
        arima_test_pred = arima_test_pred[:min_len]
        hybrid_pred = hybrid_pred[:min_len]

    # Step 4: Hybrid Final Prediction (ARIMA + LSTM Residuals)
    hybrid_final_pred = arima_test_pred + hybrid_pred

    return hybrid_final_pred, hybrid_actual[:len(hybrid_final_pred)], hybrid_mae, hybrid_rmse, hybrid_model


hybrid_pred, hybrid_actual, hybrid_mae, hybrid_rmse, hybrid_model = train_hybrid(arima_model, close_prices, X_lstm, Y_lstm)

# --- 4. Compare Results ---
print("Model Comparison:")
print(f"ARIMA   -> MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}")
print(f"LSTM    -> MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}")
print(f"Hybrid  -> MAE: {hybrid_mae:.4f}, RMSE: {hybrid_rmse:.4f}")

# --- 5. Plot Results ---
plt.figure(figsize=(12,6))
plt.plot(arima_actual, label="Actual Prices", color="black", linestyle="dotted")
plt.plot(arima_pred, label="ARIMA Predictions", linestyle="dashed")
plt.plot(lstm_pred, label="LSTM Predictions", linestyle="dashdot")
plt.plot(hybrid_pred, label="Hybrid (ARIMA+LSTM)", linestyle="solid")
plt.legend()
plt.title("Stock Price Predictions: ARIMA vs LSTM vs Hybrid")
plt.show()
