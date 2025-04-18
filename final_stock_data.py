import pyodbc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Database connection details
server = 'localhost'  
database = 'EthanStockPrice'
driver = 'ODBC Driver 17 for SQL Server'  

# Connect to SQL Server
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes')

# Query to retrieve data
query = "SELECT * FROM StockData"
data = pd.read_sql(query, conn)

# Data Cleaning
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove rows with negative or zero values in essential columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_columns:
        df = df[df[col] > 0]

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])

    return df

# Apply cleaning
cleaned_data = clean_data(data)

# Normalization
def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Apply normalization
normalized_data = normalize_data(cleaned_data)


# Missing Value Handling
def handle_missing_values(df):
    # Fill missing numeric values with the median
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Drop rows with missing dates or symbols
    df = df.dropna(subset=['Date', 'Symbol'])

    return df

# Apply missing value handling
handled_data = handle_missing_values(normalized_data)

# Save final data to a file (optional)
handled_data.to_csv("final_stock_data.csv", index=False)
