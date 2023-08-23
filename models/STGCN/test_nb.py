import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import logging

from stgcn import STGCN_NegativeBinomial
from utils import generate_dataset,accuracy, load_metr_la_data,load_crash_data, load_crime_data,get_normalized_adj,NegativeBinomialLoss,masked_mae,load_demand_data

use_gpu = False
num_timesteps_input = 12
num_timesteps_output = 12

epochs = 1000
batch_size = 24

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--best-model', type=str,
                    default='pth/origin_1.pth'.format(num_timesteps_input, num_timesteps_output, batch_size),
                    help='Directory to load the trained model')
parser.add_argument('--using-data', type=str,default='metr',help='What data to use')
parser.add_argument('--output-dir', type=str,
                    default='output/metr_in{:}_out{:}_batch{:}.npz'.format(num_timesteps_input, num_timesteps_output, batch_size),
                    help='Directory to save the outputs')

args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
device = args.device

net = torch.load(args.best_model).to(device=device)   # Load the model

# Load datasets
if args.using_data == 'metr':
    A, X, means, stds = load_metr_la_data()
elif args.using_data == 'crime_daily':
    A, X = load_crime_data('daily')
elif args.using_data == 'crash_daily':
    A, X = load_crash_data('daily')
elif args.using_data == 'crime_1w':
    A, X = load_crime_data('1w')
elif args.using_data == 'crash_1w':
    A, X = load_crash_data('1w')
elif args.using_data == 'crime_8h':
    A, X = load_crime_data('8h')
elif args.using_data == 'crash_8h':
    A, X = load_crash_data('8h')
elif args.using_data == 'crime_1h':
    A, X = load_crime_data('1h')
elif args.using_data == 'crash_1h':
    A, X = load_crash_data('1h')
elif args.using_data == 'demand_5min':
    A, X = load_demand_data('5min')
elif args.using_data == 'demand_15min':
    A, X = load_demand_data('15min')
elif args.using_data == 'demand_60min':
    A, X = load_demand_data('60min')

split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)

train_original_data = X[:, :, :split_line1]
test_original_data = X[:, :, split_line2:]

training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)

A_wave = get_normalized_adj(A)
A_wave = torch.from_numpy(A_wave)

A_wave = A_wave.to(device=args.device)
loss_criterion = NegativeBinomialLoss()

test_losses = []
test_maes = []
with torch.no_grad():
    net.eval()
    
    permutation = torch.arange(test_input.shape[0])
    
    out_mu, out_alpha = [],[]
    for i in range(0, test_input.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = test_input[indices], test_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        
        out_mu_batch, out_alpha_batch = net(A_wave, X_batch)
        out_mu.append(out_mu_batch)
        out_alpha.append(out_alpha_batch)
    # test_input = test_input.to(device=device)
    # test_target = test_target.to(device=device)
    
    # out_mu, out_alpha = net(A_wave, test_input)    
    # test_loss = loss_criterion(out_mu,out_alpha,test_target).to(device="cpu")
    # test_losses.append(test_loss.detach().numpy())
    
    # out_unnormalized = out_mu.detach().cpu().numpy()
    out_mu = torch.cat(out_mu,dim=0).detach().cpu().numpy()
    out_unnormalized = out_mu.copy()
    out_alpha = torch.cat(out_alpha,dim=0).detach().cpu().numpy()
    
    target_unnormalized = test_target.detach().cpu().numpy()
    
    # The error of each horizon
    mae_list = []
    rmse_list=[]
    mape_list=[]
    for horizon in range(out_unnormalized.shape[2]):
        mae  = np.mean(np.abs(out_unnormalized[:,:,horizon] - target_unnormalized[:,:,horizon]))
        rmse = np.sqrt(np.mean(out_unnormalized[:,:,horizon] - target_unnormalized[:,:,horizon]))  
        mape = np.mean(np.abs( (out_unnormalized[:,:,horizon] - target_unnormalized[:,:,horizon])/(target_unnormalized[:,:,horizon]+1e-5) ))
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        print('Horizon %d MAE:%.4f RMSE:%.4f MAPE:%.4f'%(horizon,mae,rmse,mape))
    print('BestModel %s overall score: mae %.4f; rmse %.4f; mape %.4f'%(args.best_model,np.mean(mae_list),np.mean(rmse_list),np.mean(mape_list)))
np.savez_compressed(args.output_dir,target=test_target.detach().cpu().numpy(),
                    out_mu=out_mu,out_alpha=out_alpha,
                    test_input=test_input.detach().cpu().numpy(),
                    out_unnormalized=out_unnormalized,target_unnormalized=target_unnormalized)