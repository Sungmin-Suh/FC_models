# data preprocessing
import pandas as pd  # 데이터 조작 및 분석 라이브러리
import numpy as np  # 다차원 배열 처리 지원 라이브러리
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# import keras.utils as utils
from tensorflow.keras.models import Sequential  # CNN 모델 만들기위한 라이브러리
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, BatchNormalization, Flatten  # CNN 모델 구성요소 라이브러리
from tensorflow.keras.optimizers import Adam  # 훈련 옵티마이저 라이브러리
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # 모델 훈련 라이브러리
from math import sqrt  # 수학 라이브러리 (root)
from sklearn.model_selection import train_test_split  # 훈련 검증 테스트 데이터 나누기 라이브러리
import tensorflow as tf  # 텐서플로우 라이브러리
import os  # Directory 설정 라이브러리

# os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
in_tot = o[:, 0:16]
out_rank = o[:, 16]

in_tot_B = o2[:, 0:16]
out_rank_B = o2[:, 16]

xTrain=in_tot
yTrain=out_rank
xVal=in_tot_B
yVal=out_rank_B

tr_size = xTrain.shape  # 훈련 데이터셋 차원 변수지정
print("Train size=", tr_size)  # 훈련 데이터셋 차원 보이기

vl_size = xVal.shape  # 검증 데이터셋 차원 변수지정
print("Test size=", vl_size)  # 검증 데이터셋 차원 보이기

tr_ss = tr_size[0]  # 훈련 데이터셋 0번째 차원 변수지정
tr_ff = tr_size[1]  # 훈련 데이터셋 1번째 차원 변수지정

vl_ss = vl_size[0]  # 검증 데이터셋 0번째 차원 변수지정
vl_ff = vl_size[1]  # 검증 데이터셋 1번째 차원 변수지정

x1Train = np.reshape(xTrain, (tr_ss, tr_ff, -1))  # reshape을 통한 훈련데이터 차원 재배치
x1Test = np.reshape(xVal, (vl_ss, vl_ff, -1))  # reshape을 통한 검증데이터 차원 재배치

print(x1Train.shape)
print(x1Test.shape)

# 함수의 파라미터에 -1이 들어가면 특별한 의미를 갖는데, 다른 나머지 차원 크기를 맞추고 남은 크기를 해당 차원에 할당해 준다는 의미입니다

y1Train = np.reshape(yTrain, (tr_ss, -1))  # reshape을 통한 훈련 라벨데이터 차원 재배치
y1Test = np.reshape(yVal, (vl_ss, -1))  # reshape을 통한 검증 라벨데이터 차원 재배치

print(y1Train.shape)
print(y1Test.shape)


xxTrain = tf.convert_to_tensor(x1Train, dtype=tf.float32)  # 훈련 데이터 텐서 변수로 전환
yyTrain = tf.convert_to_tensor(y1Train, dtype=tf.float32)  # 검증 데이터 텐서 변수로 전환
xxVal = tf.convert_to_tensor(x1Test, dtype=tf.float32)  # 훈련 라벨 데이터 텐서 변수로 전환
yyVal = tf.convert_to_tensor(y1Test, dtype=tf.float32)  # 검증 라벨 데이터 텐서 변수로 전환

sequence_length = tr_ff  # 입력 갯수 (여기선 9개)


num_features = 1
model = Sequential()
model.add(Conv1D(name="cnn1", filters=96, kernel_size=3, padding='same', strides=1, input_shape=(sequence_length, num_features)))  # 1D-CNN 첫번째 레이어 구성, 필터갯수: 64, 필터사이즈: 1x8, 패딩 적용, 스트라이드 1,길이 6
# model.add(BatchNormalization(epsilon=1e-06,momentum=0.9,  weights=None)) # 배치정규화
model.add(Activation('relu'))
# model.add(Dropout(0.2)) # 드롭 아웃 적용

model.add(Conv1D(name="cnn2", filters=128, kernel_size=3, padding='same',strides=1))  # 1D-CNN 첫번째 레이어 구성, 필터갯수: 64, 필터사이즈: 1x8, 패딩 적용, 스트라이드 1,길이 6
# model.add(BatchNormalization(epsilon=1e-06,momentum=0.9,  weights=None)) # 배치정규화
model.add(Activation('relu'))
# model.add(Dropout(0.2)) # 드롭 아웃 적용

model.add(Conv1D(name="cnn3", filters=384, kernel_size=3, padding='same',strides=1))  # 1D-CNN 첫번째 레이어 구성, 필터갯수: 64, 필터사이즈: 1x8, 패딩 적용, 스트라이드 1,길이 6
# model.add(BatchNormalization(epsilon=1e-06,momentum=0.9 ,  weights=None)) # 배치정규화
model.add(Activation('relu'))
# model.add(Dropout(0.3))
#
model.add(Conv1D(name="cnn4", filters=640, kernel_size=3, padding='same',strides=1))  # 1D-CNN 첫번째 레이어 구성, 필터갯수: 64, 필터사이즈: 1x8, 패딩 적용, 스트라이드 1,길이 6
model.add(BatchNormalization(epsilon=1e-06, momentum=0.9,  weights=None)) # 배치정규화
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation=None))

model.summary()  # 모델 구성 요약을 보여줌

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-4))  # 손실함수 평균제곱근 오차, 경사하강법 최적화 방법: adam 적용, 학습률: 0.001

batch_size = 32  # 배치사이즈 설정
epochs = 2000  # 훈련 횟수 설정

history = model.fit(xxTrain, yyTrain, batch_size=batch_size, epochs=epochs, validation_data=(xxVal, yyVal), #검증의 라벨링 데이터
                    callbacks = [ # 훈련중 다양한 설정 가능하게한 Callback 함수
                        EarlyStopping(monitor='val_loss',patience=30, verbose=1, mode='min'), # 모델 조기 종료를 위한 함수
                        ModelCheckpoint(filepath='1d_cnn_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1), # 가장 모델이 좋게 나온 모델(가중치) 저장
                        ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1,min_delta=1e-5,mode='min')]) #학습시 learning rate 자동 변경

# 학습과정 가시화
history_dict = history.history
history_dict.keys()
fig, ax1 = plt.subplots(figsize=(20, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss', fontsize=30)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend(['Train', 'Validation'], loc='upper right')

model.load_weights('1d_cnn_model.h5')
y_pred = model.predict(xxTrain)  # 훈련 모델 결과 생성
y_vali = model.predict(xxVal)  # 검증 모델 결과 생성
train_rs_f = np.ravel(y_pred)  # 다차원화된 훈련 결과 벡터화
validation_rs_f = np.ravel(y_vali)  # 다차원화된 검증 결과 벡터화

fig, ax1 = plt.subplots(figsize=(20, 8))
#
plt.rcParams['font.family'] = 'Times New Roman'

x = np.array(range(0, 10))
plt.subplot(1, 2, 1)
plt.scatter(yTrain, train_rs_f, zorder=2, color = 'black')
plt.title('Train result', fontsize=25, fontweight='bold')
plt.xlabel('Log(Observed F.coli)', fontsize=17, fontweight='bold')
plt.ylabel('Log(Estimated F.coli)', fontsize=17, fontweight='bold')
plt.plot(x, 1*x, color='red', zorder=1)
# max_value = max(max(yTrain), max(train_rs_f))
# plt.xlim(0, max_value + 0.1*max_value)
# plt.ylim(0, max_value + 0.1*max_value)

plt.subplot(1, 2, 2)
plt.scatter(yVal, validation_rs_f, zorder=2, color = 'black')
plt.title('Validation result', fontsize=25, fontweight='bold')
plt.xlabel('Log(Observed F.coli)', fontsize=17, fontweight='bold')
plt.ylabel('Log(Estimated F.coli)', fontsize=17, fontweight='bold')
plt.plot(x, color='red', zorder=1)
# max_value = max(max(yTrain), max(train_rs_f))
# plt.xlim(0, max_value + 0.1*max_value)
# plt.ylim(0, max_value + 0.1*max_value)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

def nse(predictions, targets):
    out = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    return out

tr_nse = nse(train_rs_f, yTrain)
vl_nse = nse(validation_rs_f, yVal)
print("Training NSE=", tr_nse)
print("Test NSE=", vl_nse)

tr_rmse = sqrt(mean_squared_error(yTrain, train_rs_f))
vl_rmse = sqrt(mean_squared_error(yVal, validation_rs_f))
print("Training RMSE=", tr_rmse)
print("Test RMSE=", vl_rmse)

# # Save RMSE and NSE values to a CSV
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
#
# # Save results to CSV
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