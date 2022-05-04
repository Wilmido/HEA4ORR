import xgboost as xgb
import numpy as np
from utils.evaluation import evaluate
from utils.data_loader import BasicDataset
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso

dataset = BasicDataset('data/new_data.csv')
x_train, y_train, x_test, y_test = dataset.get_data(device='cpu')

x_train = x_train.detach().numpy().reshape(len(y_train), -1)
y_train = y_train.detach().numpy()
x_test = x_test.detach().numpy().reshape(len(y_test), -1)
y_test = y_test.detach().numpy()


regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train.ravel())
y_pred = regressor.predict(x_test)
mae, rmse = evaluate(y_test, y_pred.reshape(-1, 1))
print(f'svr\nMAE: {mae} RMSE: {rmse}')


ridge =Ridge(alpha=0.01, fit_intercept=True)    
ridge.fit(x_train,y_train.ravel())
y_pred =ridge.predict(x_test) 
mae, rmse = evaluate(y_test, y_pred.reshape(-1, 1))
print(f'Ridge\nMAE: {mae} RMSE: {rmse}')


lasso =Lasso(alpha=0.01, fit_intercept=True)    
lasso.fit(x_train,y_train.ravel())
y_pred =lasso.predict(x_test) 
mae, rmse = evaluate(y_test, y_pred.reshape(-1, 1))
print(f'Lasso\nMAE: {mae} RMSE: {rmse}')


params=[1]
for param in params:
    reg = xgb.XGBRegressor(n_estimators=1000, max_depth=param)
    reg.fit(x_train, y_train)
    test_pred = reg.predict(x_test, -1)


    test_MAE, test_RMSE = evaluate(y_test, test_pred.reshape(-1, 1))
    
    print(f"param: {param} MAE: {test_MAE} RMSE: {test_RMSE}")
