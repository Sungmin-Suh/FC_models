# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import pandas as pd
import os

# os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


filename = 'train_data(san).xlsx'
df = pd.read_excel(filename, engine='openpyxl', sheet_name=None)
df = pd.concat([value.assign(sheet_source=key) for key, value in df.items()], ignore_index=True)
df = df.dropna(axis=0)

o = df.to_numpy()

filename2 = 'validation_data(san).xlsx'
df2 = pd.read_excel(filename2, engine='openpyxl', sheet_name=None)
df2 = pd.concat([value.assign(sheet_source=key) for key, value in df2.items()], ignore_index=True)
df2 = df2.dropna(axis=0)

o2 = df2.to_numpy()

# IQR을 사용하여 이상치 제거
in_tot = o[:, 0:15]
out_rank = o[:, 15]

in_tot_B = o2[:, 0:15]
out_rank_B = o2[:, 15]

xTrain=in_tot
yTrain=out_rank
xVal=in_tot_B
yVal=out_rank_B

# 입력자료 tensor 형식으로 변경
xxTrain=tf.convert_to_tensor(xTrain, dtype=tf.float32)
yyTrain=tf.convert_to_tensor(yTrain, dtype=tf.float32)
xxVal=tf.convert_to_tensor(xVal, dtype=tf.float32)
yyVal=tf.convert_to_tensor(yVal, dtype=tf.float32)

params = { 'n_estimators': [500],
           'max_depth': [5, 7],
           'learning_rate': [0.01, 0.05],
           'subsample': [0.3],
           'gamma':[0.3]}
#
# algorithm 객체 생성 후 GridSearchCV 수행
xgr_reg = xgb.XGBRegressor(objective='reg:squarederror')
grid_cv = GridSearchCV(xgr_reg, param_grid=params, cv=4, n_jobs=-1, scoring='r2')
grid_cv.fit(xTrain, yTrain)
#
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
# print(grid_cv.error_score)
scores_df = pd.DataFrame(grid_cv.cv_results_)
# print(scores_df)

model = grid_cv.best_estimator_
model.fit(xxTrain,yyTrain)

y_pred = model.predict(xxTrain) # 훈련 모델 결과 생성
y_vali = model.predict(xxVal) # 검증 모델 결과 생성
train_rs_f=np.ravel(y_pred) # 다차원화된 훈련 결과 벡터화
validation_rs_f=np.ravel(y_vali) # 다차원화된 검증 결과 벡터화

fig, ax1 = plt.subplots(figsize=(20, 8))
#
plt.rcParams['font.family'] = 'Times New Roman'

x = np.array(range(0, 21))
plt.subplot(1, 2, 1)
plt.scatter(yTrain, train_rs_f, zorder=2, color = 'black')
plt.title('Train result', fontsize=25, fontweight='bold')
plt.xlabel('Log(Observed F.coli)', fontsize=17, fontweight='bold')
plt.ylabel('Log(Estimated F.coli)', fontsize=17, fontweight='bold')
plt.plot(x, color='red', zorder=1)
max_value = max(max(yTrain), max(train_rs_f))
plt.xlim(0, max_value + 0.1*max_value)
plt.ylim(0, max_value + 0.1*max_value)

plt.subplot(1, 2, 2)
plt.scatter(yVal, validation_rs_f, zorder=2, color = 'black')
plt.title('Validation result', fontsize=25, fontweight='bold')
plt.xlabel('Log(Observed F.coli)', fontsize=17, fontweight='bold')
plt.ylabel('Log(Estimated F.coli)', fontsize=17, fontweight='bold')
plt.plot(x, color='red', zorder=1)
max_value = max(max(yTrain), max(train_rs_f))
plt.xlim(0, max_value + 0.1*max_value)
plt.ylim(0, max_value + 0.1*max_value)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

def nse(predictions, targets):
    out= 1 - (np.sum((targets- predictions)**2)/np.sum((targets-np.mean(targets))**2))
    return out

# tr_r2=r2_score(yTrain, train_rs_f)
# vl_r2=r2_score(yTest, validation_rs_f)
# print("Training R2=", tr_r2)
# print("Test R2=", vl_r2)

tr_nse = nse(train_rs_f, yTrain)
vl_nse = nse(validation_rs_f, yVal)
print("Training NSE= {:.5f}".format(tr_nse))
print("Test NSE= {:.5f}".format(vl_nse))

tr_rmse = sqrt(mean_squared_error(yTrain, train_rs_f))
vl_rmse = sqrt(mean_squared_error(yVal, validation_rs_f))
print("Training RMSE= {:.5f}".format(tr_rmse))
print("Test RMSE= {:.5f}".format(vl_rmse))

log_base = 10

# Convert natural log(RMSE) back to RMSE using exp
tr_rmse_original = np.exp(tr_rmse)
vl_rmse_original = np.exp(vl_rmse)

print("Training RMSE (original scale) = {:.5f}".format(tr_rmse_original))
print("Validation RMSE (original scale) = {:.5f}".format(vl_rmse_original))

# metrics_df = pd.DataFrame({
#     'Dataset': ['Training', 'Validation'],
#     'RMSE': [tr_rmse, vl_rmse],
#     'NSE': [tr_nse, vl_nse]
# })
# metrics_df.to_csv('metrics_results.csv', index=False)
#
# # Save the predicted and actual values for both training and validation datasets:
# train_results_df = pd.DataFrame({
#     'Actual_Train': yTrain.ravel(),
#     'Predicted_Train': train_rs_f.ravel()
# })
#
# val_results_df = pd.DataFrame({
#     'Actual_Validation': yVal.ravel(),
#     'Predicted_Validation': validation_rs_f.ravel()
# })

# Save results to CSV
# train_results_df.to_csv('train_results.csv', index=False)
# val_results_df.to_csv('validation_results.csv', index=False)

# # Save training and validation datasets:
# train_df = pd.DataFrame(xTrain, columns=df.columns[3:19])
# train_df['output'] = yTrain
# train_df.to_csv('train_data.csv', index=False)
#
# val_df = pd.DataFrame(xVal, columns=df.columns[3:19])
# val_df['Output'] = yVal
# val_df.to_csv('validation_data.csv', index=False)

# plt.show()



