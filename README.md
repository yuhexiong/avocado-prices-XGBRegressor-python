# Avocado Prices XGBRegressor

### DataSet From [Kaggle - Avocado Prices](https://www.kaggle.com/datasets/neuromusic/avocado-prices)

## Overview

- Language: Python v3.9.15
- Package: Scikit-Learn, xgboost
- Model: XGBRegressor
- Loss Function: Mean Squared Error
- Use GridSearchCV For Automatic Parameter Tuning, cv = 5
- Label Encoding to deal with string type column

## Data Distribution

![image](https://github.com/yuhexiong/avocado-prices-XGBRegressor-python/blob/main/image/data_distribution.png)

## Result

- Best Params: {'gamma': 0, 'max_depth': 7, 'n_estimators': 1000, 'reg_lambda': 10}
- Mean Squared Error: 0.03500582893385328
- R2: 0.79
