import yfinance as yf 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd 

sp = yf.Ticker('^GSPC') #gets the symbol
sp = sp.history(period = 'max') #fetching the data for the symbol
sp = sp.loc['1990-01-01':].copy()
del sp['Dividends']
del sp['Stock Splits']

sp['Tomorrow'] = sp['Close'].shift(-1 )  #creating the tomorrow column 
sp['PrevDayChange'] = sp['Close'].diff()
sp['PrevDayTrend'] = (sp['PrevDayChange'] > 0).astype(int)
sp['Target'] = (sp['Tomorrow'] > sp['Close']).astype(int) #creating the target column which is

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

predictors = ['Volume', 'High', 'Low']

def predict(train, test, predictors, model):
  model.fit(train[predictors], train['Target'])
  preds = model.predict(test[predictors])
  preds = pd.Series(preds, index=test.index, name='Predictions')
  combined = pd.concat([test['Target'], preds], axis=1)
  return combined


def backtest(data, model, predictors, start = 2500, step = 250):
  all_predictions = []
  
  for i in range(start, data.shape[0], step):
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()
    predictions = predict(train, test, predictors, model)
    all_predictions.append(predictions) 
  return pd.concat(all_predictions)

all_predictions = backtest(sp, model, predictors)
precision = precision_score(all_predictions['Target'], all_predictions['Predictions'])

print('Precision: %.3f' % precision)