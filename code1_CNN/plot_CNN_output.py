import numpy as np
import datetime
import cartopy.crs as ccrs
import scipy.io as sio
from matplotlib import pyplot as plt
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, r2_score
from CNN import p,q

mat_name = f'CNN_output_25_25_p{p}_q{q}.mat'
mats = sio.loadmat(mat_name)
CNN_pred=mats['CNN_pred']
label_test=mats['label_test']
swan_test=mats['swan_test']
x_swan=mats['x_swan']
y_swan=mats['y_swan']
x_label=mats['x_label']
y_label=mats['y_label']
times_str=mats['times_str']

# 把两者的空间网格进行匹配
pos_lat=np.arange(0,25,1)
pos_lon=np.arange(0,25,1)
x_swan=x_swan[np.ix_(pos_lat,pos_lon)]
y_swan=y_swan[np.ix_(pos_lat,pos_lon)]
swan_test=swan_test[np.ix_(np.arange(len(times_str)),pos_lat,pos_lon)] # time×lat×lon

time=[]
for i, date in enumerate(times_str):
    time.append(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))

def plot_data(sub_n,x,y,data,t_str,label):
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, sub_n, projection=proj)
    ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=proj)
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='black',
                                            facecolor=cfeature.COLORS['land'])
    cf = ax.contourf(x,y, data, transform=proj,cmap="rainbow")
    ax.add_feature(land, zorder=1)
    cb = fig.colorbar(cf, ax=ax) # ,shrink=0.6, pad=0.1  orientation='horizontal'可以将colorbar水平放置
    cb.set_label(label, rotation=0,  y=1.07,labelpad=-30,fontsize=12)
    ax.tick_params(labelsize=12)
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    # ax.set_title(t_str,fontsize=12)

    # Set colorbar range
    # cb.set_clim(vmin=-1, vmax=1)

    # Set colorbar position
    cb.ax.set_position([0.78, 0.1, 0.02, 0.75])  # left, bottom, width, height


def calculate_rmse_mae_mse_r2(matrix1, matrix2):
    # Ensure the matrices have the same shape
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"

    # Calculate the difference between the two matrices
    diff = matrix1 - matrix2

    # Flatten the difference matrix to 1D for simplicity
    diff = diff.flatten()
    matrix1_flatten = matrix1.flatten()
    matrix2_flatten = matrix2.flatten()

    # Calculate RMSE
    rmse = round(np.sqrt(np.mean(diff ** 2)), 2)

    # Calculate MAE
    mae = round(np.mean(np.abs(diff)), 2)

    # Calculate MSE
    mse = round(mean_squared_error(matrix1_flatten, matrix2_flatten), 2)

    # Calculate R^2
    r2 = round(r2_score(matrix1_flatten, matrix2_flatten), 2)

    return rmse, mae, mse, r2



rmse, mae, mse, r2 = calculate_rmse_mae_mse_r2(swan_test, label_test)
print(f"SWAN_result RMSE: {rmse:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")

rmse, mae, mse, r2 = calculate_rmse_mae_mse_r2(CNN_pred, label_test)
print(f"CNN_result　RMSE: {rmse:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")

# ssim_value = calculate_ssim(CNN_pred, label_test)
# print(f"CNN_result　SSIM: {ssim_value:.2f}")
## 将两种结果进行对比
# fig = plt.figure(figsize=(10, 6))
# n=1
# label1='SWAN-ERA5' # swan输出值减去真实值
# plot_data(1,x_swan,y_swan, swan_test[n,:,:,0]-label_test[n,:,:,0],times_str[n],label1)
# label2='SWAN&CNN-ERA5' # CNN输出值减去真实值
# plot_data(2,x_swan,y_swan, CNN_pred[n,:,:,0]-label_test[n,:,:,0],times_str[n],label2)
# filename = f'CNN_result_3month_25_25_p{p}_q{q}.png'
# plt.savefig(filename, dpi=600)
# plt.show()

#   分开绘图
fig = plt.figure(figsize=(10, 6))
n=1
label1='Hs (m)' # swan输出值减去真实值
plot_data(1,x_swan,y_swan, swan_test[n,:,:,0]-label_test[n,:,:,0],'(a) SWAN',label1)
# plt.savefig(filename, dpi=600)
plt.text(0.5, -0.1, '(a) SWAN', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.show()

fig = plt.figure(figsize=(10, 6))
label2='SWAN&CNN-ERA5' # CNN输出值减去真实值
plot_data(1,x_swan,y_swan, CNN_pred[n,:,:,0]-label_test[n,:,:,0],'(b) CNN',label1)
filename = f'CNN_result_3month_25_25_p{p}_q{q}.png'
# plt.savefig(filename, dpi=600)
plt.show()

# 绘制散点图，其中label_test为真实值。
# 此外绘图时不绘制X轴或者Y轴为0的数，最后绘制斜率直线，并在图中标注出来

# Flatten the matrices to 1D arrays
swan_test_flatten = swan_test[:,:,:,1].flatten()
label_test_flatten = label_test[:,:,:,1].flatten()


# Find indices where both arrays are not zero
indices = np.where(np.logical_and(swan_test_flatten != 0, label_test_flatten != 0))

# Use these indices to get the non-zero elements of both arrays
swan_test_flatten = swan_test_flatten[indices]
label_test_flatten = label_test_flatten[indices]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(label_test_flatten, swan_test_flatten, s=5, alpha=0.5)

# Fit a line to the data
slope, intercept = np.polyfit(label_test_flatten, swan_test_flatten, 1)

# Add the line to the plot
x = np.array([label_test_flatten.min(), label_test_flatten.max()])
y = slope * x + intercept
plt.plot(x, y, 'r', label=f'y={slope:.2f}x+{intercept:.2f}')

# Add slope annotation
plt.text(0.5, 0.7, f'Slope: {slope:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Set labels and title
plt.xlabel('Observation Hs (m)')
plt.ylabel('Prediction Hs (m)')
# plt.title('Scatter plot of model_pred vs label_test')
plt.legend(frameon=False)
filename = f'SWAN_slope_p{p}_q{q}.png'
plt.savefig(filename, dpi=600)
# Show the plot
plt.show()

# 绘制散点图，其中label_test为真实值。
# 此外绘图时不绘制X轴或者Y轴为0的数，最后绘制斜率直线，并在图中标注出来

# Flatten the matrices to 1D arrays
CNN_pred_flatten = CNN_pred[:,:,:,1].flatten()
label_test_flatten = label_test[:,:,:,1].flatten()

# Find indices where both arrays are not zero
indices = np.where(np.logical_and(CNN_pred_flatten != 0, label_test_flatten != 0))

# Use these indices to get the non-zero elements of both arrays
CNN_pred_flatten = CNN_pred_flatten[indices]
label_test_flatten = label_test_flatten[indices]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(label_test_flatten, CNN_pred_flatten, s=5, alpha=0.5)

# Fit a line to the data
slope, intercept = np.polyfit(label_test_flatten, CNN_pred_flatten, 1)

# Add the line to the plot
x = np.array([label_test_flatten.min(), label_test_flatten.max()])
y = slope * x + intercept
plt.plot(x, y, 'r', label=f'y={slope:.2f}x+{intercept:.2f}')

# Add slope annotation
plt.text(0.5, 0.7, f'Slope: {slope:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

# Set labels and title
plt.xlabel('Observation Hs (m)')
plt.ylabel('Prediction Hs (m)')
# plt.title('Scatter plot of model_pred vs label_test')
plt.legend(frameon=False)
filename = f'CNN_slope_p{p}_q{q}.png'
plt.savefig(filename, dpi=600)
# Show the plot
plt.show()