import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import argparse
import torch
import torch.nn as nn
from model import Solver
import logging
import os
from datetime import datetime
from self_dataloader import AIEarthDataset, input_train, label_train, input_test, label_test, times, Hs_swan,X,Y
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import scipy.io as sio
from sklearn.metrics import mean_squared_error, r2_score
from self_dataloader import p,q,train_num,swh,EPOCH

"""

Compare the input SWAN data through the spatio-temporal network with the output ERA5 data.
    The main parameters are p q number of input data and number of output data.    
    EPOCH training times
    are defined in self_dataloader.py

date ：2024/7/21

"""

def parse():
    parser = argparse.ArgumentParser(description='It is interesting.')
    parser.add_argument('--gpuID', default=0, type=int)
    parser.add_argument('--inLen', default=p, type=int)
    parser.add_argument('--outLen', default=q, type=int)
    parser.add_argument('--patchSize', default=64, type=int)
    parser.add_argument('--batchSize', default=12, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--type', default='SWAN', type=str)
    parser.add_argument('--testOnly', action='store_true')
    parser.add_argument('--maskType', default='WithoutMask', type=str, choices=['MaskOnly', 'WithoutMask', 'Mix'])
    parser.add_argument('--model', default='ConvGRUNet', type=str,
                        choices=['ATMGRUNet', 'ConvGRUNet', 'ConvLSTMNet', 'DeformConvGRUNet', 'TrajGRUNet', 'MIMNet',
                                 'E3DLSTMNet', 'PredRNNPP', 'PredRNNNet', 'ABModel'])
    parser.add_argument('--abChoice', default='', type=str)
    args = parser.parse_args()
    return args


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=1e-3):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

    def forward(self, input, target):
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum((input - target) ** 2, (2, 3, 4))
        mae = torch.sum(torch.abs((input - target)), (2, 3, 4))

        mse = torch.sum(torch.mean(mse, dim=1))
        mae = torch.sum(torch.mean(mae, dim=1))
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight * mse + self.mae_weight * mae)

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predicted, target):
        return torch.mean(torch.abs(predicted - target))

def plot_data(sub_n,x,y,data,t_str,label):
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 2, sub_n, projection=proj)
    ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=proj)
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='black',
                                            facecolor=cfeature.COLORS['land'])
    cf = ax.contourf(x,y, data, transform=proj,cmap="rainbow")
    ax.add_feature(land, zorder=1)
    cb = fig.colorbar(cf, ax=ax,shrink=0.7, pad=0.1) #  orientation='horizontal'可以将colorbar水平放置
    cb.set_label(label, rotation=0,  y=1.07,labelpad=-30,fontsize=12)
    ax.tick_params(labelsize=12)
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(t_str,fontsize=12)

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

if __name__ == '__main__':
    args = parse()
    device = torch.device('cuda:%s' % args.gpuID)
    inLen, outLen, patchSize, batchSize, lr = args.inLen, args.outLen, args.patchSize, args.batchSize, args.lr
    comment = '%s%s%s%s_%s-in_%s-out_lr-%s' % (args.type, args.maskType, args.model, args.abChoice, inLen, outLen, lr)
    args.device, args.comment, args.criterion = device, comment, Weighted_mse_mae()
    # args.device, args.comment, args.criterion = device, comment, MAELoss()

    writer = None #SummaryWriter(comment=comment, logdir=args.type+args.maskType) if not args.testOnly else None
    print('Use GPU:%s' % device)

    trainset = AIEarthDataset(input_train, label_train)
    trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
    validset = AIEarthDataset(input_test, label_test)
    validLoader = DataLoader(validset, batch_size=batchSize, shuffle=True)

    validset = AIEarthDataset(input_test, label_test)
    testLoader = DataLoader(validset, batch_size=batchSize, shuffle=True)

    args.loader = [trainLoader, validLoader, testLoader]

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log = get_logger('./runs%s/%s_%s.log' % (args.type+args.maskType, 'Train' if not args.testOnly else 'Test', current_time + '_' + comment))
    args.log = [log, writer]
    solver = Solver(args)

    baseline = 999999

    for epoch in range(EPOCH):
        solver.scheduler.step(epoch)
        if not args.testOnly:
            trainLoss = solver.train(epoch)
            print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, trainLoss / len(trainLoader)))
            validLoss = solver.valid(epoch)
            print("===>Epoch {}, Average Loss: {:.4f}".format(epoch, validLoss / len(validLoader)))

    checkpoint = torch.load('./checkpoint/model_best.pth')
    solver.model.load_state_dict(checkpoint['state_dict'])
    solver.model.eval()
    solver.model.to(device)
    #   pred and plot
    input_test_tensor = torch.tensor(input_test, dtype=torch.float32)
    input_test_tensor = input_test_tensor.permute((0, 3, 1, 2))
    input_test_tensor = torch.unsqueeze(input_test_tensor, dim=2)
    input_test_tensor = input_test_tensor.to(device)
    model_pred = solver.model(input_test_tensor)       # dataxtimex1xlat×lon

    model_pred = model_pred.to("cpu")
    model_pred = torch.squeeze(model_pred, dim=2)
    model_pred_array = model_pred.detach().numpy()
    model_pred_array = np.transpose(model_pred_array, axes=[0, 2, 3, 1])

    # Extracting data for q consecutive moments of the last test set number in the test set
    times_pred_q = [times[i:i + q] for i in range(train_num + p, swh.shape[2] - q + 1, 1)]
    # Extracting the last test set number in the test set
    swan_test_last = [Hs_swan[:, :, i:i + q] for i in range(train_num + p, swh.shape[2] - q + 1, 1)]
    swan_test = np.stack(swan_test_last)

    times_pred = times[-model_pred.shape[0]:]

    x_swan = X
    y_swan = Y

    time_str = []
    for i in times_pred:
        time_str.append(i.strftime('%Y-%m-%d %H:%M:%S'))

    rmse, mae, mse, r2 = calculate_rmse_mae_mse_r2(swan_test, label_test)
    print(f"SWAN_result RMSE: {rmse}, MAE: {mae}, MSE: {mse}, R^2: {r2}")

    rmse, mae, mse, r2 = calculate_rmse_mae_mse_r2(model_pred_array, label_test)
    print(f"{args.model}_result　RMSE: {rmse}, MAE: {mae}, MSE: {mse}, R^2: {r2}")

    fig = plt.figure(figsize=(10, 6))

    n = 1
    label1 = 'SWAN-ERA5'
    plot_data(1, x_swan, y_swan, swan_test[n, :, :,0] - label_test[n, :, :,0], time_str[n], label1)
    label2 = f'SWAN&{args.model}-ERA5'  # CNN输出值减去真实值
    plot_data(2, x_swan, y_swan, model_pred_array[n, :, :,0] - label_test[n, :, :,0], time_str[n], label2)
    filename = f'{args.model}_result_25_25_p{p}_q{q}.png'
    plt.savefig(filename, dpi=600)
    plt.show()

    # Save as mat file Plotting with matlab
    var_dict = {f'{args.model}_output': model_pred_array, 'swan_test': swan_test, 'label_test': label_test,
                'x_swan': x_swan, 'y_swan': y_swan,
                'p': p, 'q': q, 'times_str': time_str}
    filename = f'{args.model}_output_25_25_p{p}_q{q}.mat'
    sio.savemat(filename, var_dict)

    # Plot a scatter plot where label_test is the true value.
    # Additionally plot without plotting numbers with 0 on the x-axis or y-axis,
    # finally plot the slope straight line and label it in the plot

    # Flatten the matrices to 1D arrays
    model_pred_flatten = model_pred_array[:, :, :, 1].flatten()
    label_test_flatten = label_test[:, :, :, 1].flatten()

    # Find indices where both arrays are not zero
    indices = np.where(np.logical_and(model_pred_flatten != 0, label_test_flatten != 0))

    # Use these indices to get the non-zero elements of both arrays
    model_pred_flatten = model_pred_flatten[indices]
    label_test_flatten = label_test_flatten[indices]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(label_test_flatten, model_pred_flatten, s=5, alpha=0.5)

    # Fit a line to the data
    slope, intercept = np.polyfit(label_test_flatten, model_pred_flatten, 1)

    # Add the line to the plot
    x = np.array([label_test_flatten.min(), label_test_flatten.max()])
    y = slope * x + intercept
    plt.plot(x, y, 'r', label=f'y={slope:.2f}x+{intercept:.2f}')

    # Add slope annotation
    plt.text(0.5, 0.7, f'Slope: {slope:.2f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)

    # Set labels and title
    plt.xlabel('Observation Hs (m)')
    plt.ylabel('Prediction Hs (m)')
    # plt.title('Scatter plot of model_pred vs label_test')
    # plt.text(0.5, -0.3, f'(c){args.model}', horizontalalignment='center', verticalalignment='center',
    #          transform=plt.gca().transAxes)
    plt.legend(frameon=False)
    filename = f'{args.model}_slope_p{p}_q{q}.png'
    plt.savefig(filename, dpi=600)
    # Show the plot
    plt.show()