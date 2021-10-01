import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from time import time
from tqdm import tqdm
start = time()

print("read csv start")
train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
sample = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
print("read csv finish")

X = train.drop(['pressure'], axis=1)
y = train['pressure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.1,
    'subsample': 0.95,
    'colsample_bytree': 0.11,
    'max_depth': 2,
    'booster': 'gbtree',
    'reg_lambda': 66.1,
    'reg_alpha': 15.9,
    'random_state':42,
    'tree_method':'gpu_hist',
    'gpu_id':0,
    'predictor':'gpu_predictor'
}

#model = XGBRegressor(**xgb_params)
#model = XGBRegressor(n_estimators= 5000,learning_rate= 0.1,random_state=1, tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor")
model1 = LinearRegression()
model2 = RandomForestRegressor(criterion="mae")
model3 = XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0, predictor="gpu_predictor")
models = [model2, model3]
model1_rate = 0.5
model2_rate = 0.3
model3_rate = 1.0
models_rate = [model2_rate, model3_rate]

print("training start")

predicted = np.zeros((4024000,))
for i in tqdm(range(len(models))):
    print("1")
    models[i].fit(X_train,y_train)
    print("2")
    models[i].score(X_test, y_test)
    print("3")

    predicted += models[i].predict(test) * models_rate[i]


print("training finish")
predicted_pressure = pd.DataFrame({'pressure': predicted[:]})
test_result = test
test_result['pressure'] = predicted_pressure







submit_result = test_result[['id','pressure']]

submit_result.to_csv('submission.csv', index=False)

print(start-time())