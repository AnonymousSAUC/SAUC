import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import logging

from stgcn import STGCN
from utils import generate_dataset, load_metr_la_data,load_crime_data, load_crash_data, get_normalized_adj,load_demand_data


use_gpu = False
num_timesteps_input = 12
num_timesteps_output = 12

epochs = 1000
batch_size = 24

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('--data-dir', type=str,
                    default='pth/metr_in{:}_out{:}_batch{:}.pth'.format(num_timesteps_input, num_timesteps_output, batch_size),
                    help='Directory to save the best model')
parser.add_argument('--using-data', type=str,default='metr',help='What data to use')                    
parser.add_argument('--log-dir', type=str,
                    default='log/origin.txt')
parser.add_argument('--seed', type=int,
                    default=7)
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
# Set loggers
logger = logging.getLogger('Model training')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(args.log_dir,'w')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
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
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # # Run validation
        # with torch.no_grad():
        #     net.eval()
        #     val_input = val_input.to(device=args.device)
        #     val_target = val_target.to(device=args.device)

        #     out = net(A_wave, val_input)
        #     val_loss = loss_criterion(out, val_target).to(device="cpu")
        #     validation_losses.append(np.asscalar(val_loss.detach().numpy()))

        #     out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
        #     target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
        #     mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        #     validation_maes.append(mae)

        #     out = None
        #     val_input = val_input.to(device="cpu")
        #     val_target = val_target.to(device="cpu")

        # print("Epoch: {}, training loss: {}".format(epoch,training_losses[-1]))
        # print("Validation loss: {}".format(validation_losses[-1]))
        # print("Validation MAE: {}".format(validation_maes[-1]))
        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()
        logger.info("Epoch: {}, training loss: {}".format(epoch,training_losses[-1]))
        handler.flush()
        
        if loss == min(training_losses):
            best_model = copy.deepcopy(net.state_dict())
        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)

    net.load_state_dict(best_model)
    torch.save(net,args.data_dir)