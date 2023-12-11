import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

avocado = pd.read_csv('data/avocado.csv')
avocado = avocado.drop('Unnamed: 0', axis = 1)
print(avocado.head())

print("avocado.shape", avocado.shape)
print("sum(avocado.isnull().sum())", sum(avocado.isnull().sum()))

avocado['Date'] = pd.to_datetime(avocado['Date'])

# label encoder
for col in ['type', 'region']:
    le = LabelEncoder()
    le.fit(avocado[str(col)])
    avocado[str(col)] = le.transform(avocado[str(col)])

print(avocado.head())

# data distribution
fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(30, 40))
colNames = avocado.drop(columns=['AveragePrice']).columns

n = 0
for row in range(6):
    for col in range(2):
        colName = colNames[n]
        ax[row][col].scatter(avocado[colName], avocado['AveragePrice'], color='green', s=1)
        n += 1
plt.show()

y = avocado['AveragePrice']
X = avocado.drop(columns=['Date', 'region', 'AveragePrice']).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

# model
xgbRegressor = xgb.XGBRegressor()
params = {'n_estimators':[1000],'reg_lambda':[1, 3, 5, 10],'gamma':[0],'max_depth':[2, 3, 5, 7]}
scoring = ['r2','neg_mean_squared_error']
grid_kn = GridSearchCV(estimator = xgbRegressor,
                        param_grid = params,
                        cv = 5,
                        scoring = scoring,
                        refit = 'neg_mean_squared_error')

grid_kn.fit(X_train, y_train)
print("grid_kn.best_score_", grid_kn.best_score_)
print("grid_kn.best_params_", grid_kn.best_params_)

y_pred = grid_kn.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2: {round(r2_score(y_test, y_pred), 2)}")