import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import pickle

def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

def load_crime_data(resolution='daily'):
    if not os.path.isfile(f"../../datasets/chicago_crime/crime_count_comm_{resolution}.npy") or not os.path.isfile("../../datasets/chicago_crime/adjacency_matrix.pkl"):
        raise ValueError("Dataset not found!")
    with open("../../datasets/chicago_crime/adjacency_matrix.pkl", "rb") as f:
        A = pickle.load(f)
    X = np.load(f"../../datasets/chicago_crime/crime_count_comm_{resolution}.npy")
    
    X = X.astype(np.float32) # Even though our data should be integers here
    A = A.astype(np.float32)
    return A, X
        
        
def load_crash_data(resolution='daily'):
    if not os.path.isfile(f"../../datasets/chicago_crash/crash_count_police_{resolution}.npy") or not os.path.isfile("../../datasets/chicago_crash/adjacency_matrix.pkl"):
        raise ValueError("Dataset not found!")
    with open("../../datasets/chicago_crash/adjacency_matrix.pkl", "rb") as f:
        A = pickle.load(f)
    X = np.load(f"../../datasets/chicago_crash/crash_count_police_{resolution}.npy")
    
    X = X.astype(np.float32) # Even though our data should be integers here
    A = A.astype(np.float32)

    return A, X

def load_demand_data(resolution='15min'):
    if not os.path.isfile(f"../../datasets/sld_travel_demand/ny_data_{resolution}.npy") or not os.path.isfile("../../datasets/sld_travel_demand/adj.npy"):
        raise ValueError("Dataset not found!")
    
    X = np.load(f"../../datasets/sld_travel_demand/ny_data_{resolution}.npy")
    A = np.load(f"../../datasets/sld_travel_demand/adj.npy")
    
    X = X.astype(np.float32) # Even though our data should be integers here
    A = A.astype(np.float32)
    
    return A, X

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def gauss_loss(y,loc,scale,y_mask=None):
    """
    The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
    http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    """
    
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 # ugly define pi value in torch format
    LL = -1/2 * torch.log(2*torch.pi*torch.pow(scale,2)) - 1/2*( torch.pow(y-loc,2)/torch.pow(scale,2) )
    return -torch.sum(LL)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def accuracy(pred, target):
    # Calculate accuracy of prediction between two tensors
    pred = pred.argmax(dim=-1)  # get the index of the max log-probability
    correct = pred.eq(target).sum().item()  # count the number of equal elements
    total = target.numel()  # get the total number of elements
    return correct / total  # calculate accuracy

# class NegativeBinomialLoss(nn.Module):
#     def __init__(self):
#         super(NegativeBinomialLoss, self).__init__()

#     def forward(self, mu, alpha, target):
#         epsilon = 1e-10
#         alpha_stable = alpha + epsilon
#         mu_stable = mu + epsilon

#         part1 = - target * torch.log(mu_stable / (mu_stable + alpha_stable))
#         part2 = - torch.lgamma(target + alpha_stable)
#         part3 = torch.lgamma(target + 1.0)
#         part4 = torch.lgamma(alpha_stable)
#         part5 = - alpha_stable * torch.log(alpha_stable / (mu_stable + alpha_stable))

#         loss = part1 + part2 + part3 + part4 + part5

#         return loss.mean()  # Take mean over batch size

# class NegativeBinomialLoss(nn.Module):
#     def __init__(self):
#         super(NegativeBinomialLoss, self).__init__()

#     def forward(self, mu, theta, target):
#         eps = 1e-10  # to avoid log(0)
#         gamma_term = torch.lgamma(target + 1/theta) - torch.lgamma(target+1) - torch.lgamma(1/theta)
#         log_prob = gamma_term + 1/theta * (torch.log(theta) - torch.log(theta * mu + eps)) + target * (torch.log(theta * mu) - torch.log(theta * mu + eps))
#         return -torch.mean(log_prob)  # return negative log likelihood


class NegativeBinomialLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(NegativeBinomialLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, mu, alpha, target):
        epsilon = 1e-10
        alpha_stable = alpha + epsilon
        mu_stable = mu + epsilon

        part1 = - target * torch.log(mu_stable / (mu_stable + alpha_stable))
        part2 = - torch.lgamma(target + alpha_stable)
        part3 = torch.lgamma(target + 1.0)
        part4 = torch.lgamma(alpha_stable)
        part5 = - alpha_stable * torch.log(alpha_stable / (mu_stable + alpha_stable))

        loss = part1 + part2 + part3 + part4 + part5

        # L2 regularization
        reg_term = self.lambda_reg * torch.sum(alpha_stable**2)
        
        return loss.mean() + reg_term  # Take mean over batch size and add regularization term

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.nb_loss = NegativeBinomialLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, mu, alpha, target):
        nb_loss = self.nb_loss(mu, alpha, target)
        mse_loss = self.mse_loss(mu, target)

        return nb_loss + mse_loss  # You could also add weights to each term, if you want to
