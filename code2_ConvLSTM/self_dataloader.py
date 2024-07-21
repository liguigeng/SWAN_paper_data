import netCDF4
import numpy as np
import datetime
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch
from tensorflow import keras
import tensorflow as tf

EPOCH = 5
# 设置用连续p个数据去预测q小时后的数据
p = 24
q = 3

# 三个月数据 总长 2208
# swan_hs_3month_for_net_25_25.mat
# ERA5_wind_hs_25_25_3month.nc

#一个月数据  总长 720
# swan_hs_for_net_25_25.mat
# ERA5_wind_hs_25_25.nc

# 读取MAT文件 做为训练集和测试集的input 维度是lat×lon×time 9*9*720
mat_data = sio.loadmat('..\data\swan_hs_for_net_25_25.mat')
# mat_data = sio.loadmat('..\data\swan_hs_3month_for_net_25_25.mat')
X = mat_data['X']
Y = mat_data['Y']
Hs_swan = mat_data['Hs']  # 维度是lat×lon×time
Hs_swan[np.isnan(Hs_swan)] = 0
# 读取netcdf文件 做为训练集和测试集的 label   9*9*720
nc = netCDF4.Dataset('..\data\ERA5_wind_hs_25_25.nc')
# nc = netCDF4.Dataset('..\data\ERA5_wind_hs_25_25_3month.nc')
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

# 设置用连续p个数据去预测q小时后的数据
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
# input_train = torch.unsqueeze(input_train, dim=2)

# 构造数据管道
class AIEarthDataset(Dataset):
    def __init__(self, data, label):
        tmp_data = torch.tensor(data, dtype=torch.float32)
        tmp_data = tmp_data.permute((0, 3, 1, 2))
        tmp_data = torch.unsqueeze(tmp_data, dim=2)
        self.data = tmp_data
        tmp_label = torch.tensor(label, dtype=torch.float32)
        tmp_label = tmp_label.permute((0, 3, 1, 2))
        tmp_label = torch.unsqueeze(tmp_label, dim=2)
        self.label = tmp_label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

batch_size = 12
trainset = AIEarthDataset(input_train, label_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = AIEarthDataset(input_test, label_test)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)