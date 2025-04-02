import yfinance as yf
import pyodbc
import pandas as pd
from datetime import datetime, timedelta

# Define stock symbols
symbols = ['7230.KL', '5080.KL']

# Get the date range for the last two years
end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)

# Retrieve stock data
def fetch_stock_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data['Symbol'] = symbol
        data[symbol] = stock_data
    return data

stock_data = fetch_stock_data(symbols, start_date, end_date)

# Combine all data into a single DataFrame
all_data = pd.concat(stock_data.values())

# Local SQL Server connection parameters
server = 'localhost'  
database = 'EthanStockPrice'
driver = 'ODBC Driver 17 for SQL Server'  

# Connect to local SQL Server
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes')
cursor = conn.cursor()

# Create table if not exists
table_creation_query = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='StockData' AND xtype='U')
CREATE TABLE StockData (
    Date DATE,
    [Open] DECIMAL(10, 3),
    [High] DECIMAL(10, 3),
    [Low] DECIMAL(10, 3),
    [Close] DECIMAL(10, 3),
    AdjClose DECIMAL(10, 3),
    Volume BIGINT,
    Symbol NVARCHAR(50)
)
"""
cursor.execute(table_creation_query)
conn.commit()

# Insert data into the database
insert_query = """
INSERT INTO StockData ([Date], [Open], [High], [Low], [Close], AdjClose, Volume, Symbol)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

for index, row in all_data.iterrows():
    cursor.execute(insert_query, index.date(), row['Open'], row['High'], row['Low'], row['Close'], row['Adj Close'], row['Volume'], row['Symbol'])

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("Data has been successfully stored in the local SQL Server database.")
