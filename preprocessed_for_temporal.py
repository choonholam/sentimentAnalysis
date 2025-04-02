import pyodbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Database connection details
server = 'localhost'  
database = 'EthanStockPrice'
driver = 'ODBC Driver 17 for SQL Server'  

# Connect to SQL Server
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes')

# Query to retrieve data
query = "SELECT * FROM StockData ORDER BY Date"
data = pd.read_sql(query, conn)

# Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_columns:
        df = df[df[col] > 0]
    
    return df

# Apply cleaning
cleaned_data = clean_data(data)

# Missing Value Handling
def handle_missing_values(df):
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=['Date', 'Symbol'])
    return df

# Apply missing value handling
handled_data = handle_missing_values(cleaned_data)

# Normalize Data
def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler

# Apply normalization
normalized_data, scaler = normalize_data(handled_data)

# --- Create Lag Features for ARIMA ---
def create_lag_features(df, column, lags=5):
    """
    Creates lag features for ARIMA by shifting the target column by `lags` time steps.
    """
    for lag in range(1, lags + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    df.dropna(inplace=True)  # Drop rows with NaN values due to shifting
    return df

# Apply lag features for ARIMA
arima_data = create_lag_features(normalized_data.copy(), column="Close", lags=5)

# --- Create Sliding Windows for LSTM ---
def create_sliding_window(df, target_column="Close", window_size=10):
    """
    Creates sliding window sequences for LSTM training.
    Returns X (features) and Y (target values).
    """
    X, Y = [], []
    data = df[target_column].values  # Convert target column to numpy array

    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])  # Past `window_size` values as input
        Y.append(data[i + window_size])      # Next value as target

    return np.array(X), np.array(Y)

# Apply sliding window for LSTM
X_lstm, Y_lstm = create_sliding_window(normalized_data.copy(), target_column="Close", window_size=10)

# Save ARIMA and LSTM data to CSV (optional)
arima_data.to_csv("arima_stock_data.csv", index=False)
np.save("lstm_X.npy", X_lstm)  # Save LSTM X data
np.save("lstm_Y.npy", Y_lstm)  # Save LSTM Y data

print("Preprocessing complete! ARIMA and LSTM data prepared.")
