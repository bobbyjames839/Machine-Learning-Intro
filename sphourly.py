import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

sp = yf.Ticker('^GSPC')
sp_hourly = sp.history(period='730d', interval='1h')
sp_daily = sp.history(period='730d')

del sp_hourly['Dividends']
del sp_hourly['Stock Splits']
del sp_daily['Dividends']
del sp_daily['Stock Splits']

# Extract date and time from the hourly data index
sp_hourly['Date'] = sp_hourly.index.date
sp_hourly['Time'] = sp_hourly.index.time

sp_hourly = sp_hourly.groupby('Date').apply(lambda x: x[x['Time'] == x['Time'].max()]) # Filter to get the last hour of each day
sp_hourly.reset_index(drop=True, inplace=True) # Reset index after groupby operation
sp_hourly['LastHourChange'] = sp_hourly['Close'].diff() # Calculate the change in the last hour of trading

sp_hourly['LastHourTrend'] = (sp_hourly['LastHourChange'] > 0).astype(int) # Create a binary variable that is 1 if 'LastHourChange' is positive and 0 otherwise
sp_daily = sp_daily.merge(sp_hourly[['Date', 'LastHourTrend']], left_on=sp_daily.index.date, right_on='Date', how='left')

sp_daily['Tomorrow'] = sp_daily['Close'].shift(-1 )  #creating the tomorrow column 
sp_daily['Target'] = (sp_daily['Tomorrow'] > sp_daily['Close']).astype(int) #creating the target column which is


model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

predictors = ['LastHourTrend']

train = sp_daily.iloc[:-100]
test = sp_daily.iloc[-100:]
model.fit(train[predictors], train['Target'])

predictions = model.predict(test[predictors])
precision = precision_score(test['Target'], predictions)
print('Precision: %.3f' % precision)

