import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import yfinance as yf
import matplotlib.pyplot as plt

ticker_symbols = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH']

start_year = '2022-02-02'

def calculate_features(data):
    data['Normalized'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)
    data['Volume'] = data['Volume']
    data['3_day_reg'] = data['Close'].rolling(window=3).apply(lambda x: np.polyfit(range(3), x, 1)[0], raw=True)
    data['5_day_reg'] = data['Close'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    data['10_day_reg'] = data['Close'].rolling(window=10).apply(lambda x: np.polyfit(range(10), x, 1)[0], raw=True)
    data['20_day_reg'] = data['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(20), x, 1)[0], raw=True)
    return data.dropna()

accuracy_scores = []

for ticker_symbol in ticker_symbols:
    stock_data = yf.Ticker(ticker_symbol).history(start=start_year, interval='1h')
    stock_data = calculate_features(stock_data)
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    stock_data.dropna(inplace=True)
    
    stock_data['Pct_change'] = stock_data['Close'].pct_change()
    stock_data['Local_Max'] = (stock_data['Pct_change'] > 0.01) & (stock_data['Pct_change'].shift(-1) < -0.01)
    stock_data['Local_Min'] = (stock_data['Pct_change'] < -0.01) & (stock_data['Pct_change'].shift(-1) > 0.01)
    stock_data['Target'] = stock_data['Local_Max'].astype(int) - stock_data['Local_Min'].astype(int)
    
    X = stock_data[['Normalized', 'Volume', '3_day_reg', '5_day_reg', '10_day_reg', '20_day_reg']]
    y = stock_data['Target']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Stock: {ticker_symbol}, Model Accuracy: {accuracy}')
    accuracy_scores.append(accuracy)

average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f'Average Model Accuracy: {average_accuracy}')
# After calculating the 'Target' variable
print(stock_data['Target'].value_counts(normalize=True))