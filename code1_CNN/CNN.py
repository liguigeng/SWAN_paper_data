import netCDF4
import numpy as np
import datetime
import scipy.io as sio
from tensorflow import keras
import tensorflow as tf

# 设置用连续p个数据去预测q小时后的数据
p = 24
q = 3

# 读取MAT文件
mat_data = sio.loadmat('..\data\swan_hs_for_net_25_25.mat')
X = mat_data['X']
Y = mat_data['Y']
Hs_swan = mat_data['Hs']  # 维度是lat×lon×time
Hs_swan[np.isnan(Hs_swan)] = 0
# 读取netcdf文件
nc = netCDF4.Dataset('..\data\ERA5_wind_hs_25_25.nc')
# nc = netCDF4.Dataset('ERA5_wind_hs.nc')
swh = nc.variables['swh'][:]  # 维度为time×lat×lon
swh[swh < 0] = 0
swh = np.transpose(swh, axes=[1, 2, 0])  # 维度变为lat×lon×time
lon = nc.variables['longitude'][:]
lat = nc.variables['latitude'][:]
lat = lat[::-1]
swh = swh[::-1, :, :]
time = nc.variables['time']
units = time.units  # 获取time变量的起始时间
start_date = datetime.datetime.strptime(units, "hours since 1900-%m-%d %H:%M:%S.0")
times = [start_date + datetime.timedelta(days=float(t) / 24.0) for t in time]
time_str = [t.strftime('%Y-%m-%d:%H:%M:%S') for t in times]  # 转换时间格式，变成更直观的形式

train_rate = 0.92
lat_len1, lon_len1, time_len = Hs_swan.shape
lat_len2, lon_len2, _ = swh.shape
train_num = round(train_rate * time_len)
test_num = time_len - train_num


def splits_sets(var, train_num):
    var_train = var[:, :, 0:train_num]
    var_test = var[:, :, train_num:]
    return var_train, var_test


swan1, swan2 = splits_sets(Hs_swan, train_num)
swh1, swh2 = splits_sets(swh, train_num)


Length_T = train_num - (p + q) + 1
Length_M = test_num - (p + q) + 1
input_train = np.zeros((Length_T, lat_len1, lon_len1, p))
label_train = np.zeros((Length_T, lat_len2, lon_len2, q))
input_test = np.zeros((Length_M, lat_len1, lon_len1, p))
label_test = np.zeros((Length_M, lat_len2, lon_len2, q))

for i in range(Length_T):
    input_train[i] = swan1[:, :, i:i + p]
    label_train[i] = swh1[:, :, i + p:i + p + q]

for i in range(Length_M):
    input_test[i] = swan2[:, :, i:i + p]
    label_test[i] = swh2[:, :, i + p:i + p + q]

inputs = keras.layers.Input(shape=(lat_len1, lon_len1, p), name='input')
medium1 = keras.layers.Conv2D(filters=2 * p, kernel_size=(3, 3), strides=1,
                              activation='relu', padding='same')(inputs)
medium2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                              padding='same')(medium1)
out = keras.layers.Dense(units=q)(medium2)
model = keras.Model(inputs=inputs, outputs=out)
model.summary()

# 编译模型，确定optimizer、loss_function
learning_rate = 0.008
batch_size = 12

model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), loss='mae', metrics=['mae'])
# 定义早停、模型保存、学习率衰减
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                  patience=60, verbose=2, mode='min', restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                                                 verbose=1, mode='min', min_delta=0.00001, cooldown=0)
input_train = input_train.astype(np.float32)
label_train = label_train.astype(np.float32)
input_test = input_test.astype(np.float32)
label_test = label_test.astype(np.float32)

# 正式训练
model.fit(input_train, label_train, batch_size=batch_size, epochs=n_epoch, validation_data=(input_test, label_test),
          callbacks=[early_stopping, reduce_lr], verbose=2)
# 输出结果
CNN_pred = model.predict(input_test)  # 维度是time×lat×lon×1
times_pred = times[-CNN_pred.shape[0]-q +1:-q +1]
# 提取测试集中   最后测试集数量的连续3个时刻的数据
# timesss_pred = [times[i:i+3] for i in range(516, swh.shape[2]-2, 1)]
# 提取测试集中   最后测试集数量的连续q个时刻的数据
times_pred_q = [times[i:i+q] for i in range(train_num + p, swh.shape[2]-q +1 , 1)]

# swan_test = Hs_swan[:, :, -CNN_pred.shape[0]:]  # 维度是lat×lon×time
# 提取测试集中   SWAN模拟数据输出的   最后几个时刻的数据
swan_test_last = [Hs_swan[:, :, i:i+q] for i in range(train_num + p, swh.shape[2]-q +1, 1)]
swan_test = np.stack(swan_test_last)
# swan_test = np.transpose(swan_test_last_202_ndarray, axes=[2, 0, 1])  # 维度变为time×lat×lon
x_swan = X
y_swan = Y
[x_label, y_label] = np.meshgrid(np.array(lon), np.array(lat))
# 提取测试集中   SWAN模拟数据输出的   最后三个时刻的数据
# Hs_swan_last_202 = [Hs_swan[:, :, i:i+3] for i in range(516, time_len, 1)]

# 保存结果
time_str = []
for i in times_pred:
    time_str.append(i.strftime('%Y-%m-%d %H:%M:%S'))  # sio.savemat无法直接保存datetime.time类

var_dict = {'CNN_pred': CNN_pred, 'swan_test': swan_test, 'label_test': label_test,
            'x_swan': x_swan, 'y_swan': y_swan, 'x_label': x_label, 'y_label': y_label,
            'p': p, 'q': q,'times_str': time_str}
# sio.savemat('CNN_output_25_25.mat', var_dict)
filename = f'CNN_output_25_25_p{p}_q{q}.mat'
# filename = f'CNN_output_3month_25_25_p{p}_q{q}.mat'
sio.savemat(filename, var_dict)


