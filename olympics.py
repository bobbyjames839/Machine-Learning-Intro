import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

teams = pd.read_csv("https://raw.githubusercontent.com/dataquestio/project-walkthroughs/master/beginner_ml/teams.csv")

teams = teams[['team', 'country', 'year', 'athletes', 'age', 'prev_medals', 'medals']]

'''sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None)

teams.plot.hist(y='medals')

plt.show()'''

teams[teams.isnull().any(axis=1)]
teams = teams.dropna()

train = teams[teams['year'] < 2012].copy()
test = teams[teams['year'] >= 2012].copy()

print(train.shape)
print(test.shape)

reg = LinearRegression()
predictors = ['athletes', 'prev_medals']
target = 'medals'

reg.fit(train[predictors], train['medals'])

predictions = reg.predict(test[predictors])
test['predictions'] = predictions
test.loc[test['predictions'] < 0, 'predictions'] = 0
test['predictions'] = test['predictions'].round()

#print(test)

error = mean_absolute_error(test['medals'], test['predictions'])

teams.describe()['medals']

print(test[test['team'] == 'USA'])
print(test[test['team'] == 'IND'])

errors = (test['medals'] - test['predictions']).abs()

error_by_team = errors.groupby(test['team']).mean()

print(error_by_team)

medals_by_team = test['medals'].groupby(test['team']).mean()

error_ratio = error_by_team / medals_by_team
error_ratio = error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]

print(error_ratio)