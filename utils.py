import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from scipy.stats import nbinom
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from scipy.integrate import quad


# Define the accuracy and calibration metrics
def accuracy_softmax(pred, target):
    # Calculate accuracy of prediction between two tensors
    pred = pred.argmax(dim=-1)  # get the index of the max log-probability
    correct = pred.eq(target).sum().item()  # count the number of equal elements
    total = target.numel()  # get the total number of elements
    return correct / total  # calculate accuracy

def accuracy_numerical(pred,target):
    correct = pred.eq(target).sum().item()  # count the number of equal elements
    total = target.numel()  # get the total number of elements
    return correct / total  # calculate accuracy

def masked_mse(true, pred, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(true)
    else:
        mask = (true != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask[np.isnan(mask)] = 0.0
    loss = (pred - true) ** 2
    loss = loss * mask
    loss[np.isnan(loss)] = 0.0
    return np.mean(loss)

def masked_rmse(true, pred, null_val=np.nan):
    return np.sqrt(masked_mse(true, pred, null_val))

def masked_mae(true, pred, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(true)
    else:
        mask = (true != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask[np.isnan(mask)] = 0.0
    loss = np.abs(pred - true)
    loss = loss * mask
    loss[np.isnan(loss)] = 0.0
    return np.mean(loss)

def masked_mape(true, pred, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(true)
    else:
        mask = (true != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask[np.isnan(mask)] = 0.0
    loss = np.abs(pred - true) / true
    loss = loss * mask
    loss[np.isnan(loss)] = 0.0
    return np.mean(loss)

def print_error(softmax_pred,orign_pred,true,description=''):
    print('===>',description)
    print('-'*10,'Softmax')
    print('Accuracy:',accuracy_softmax(torch.Tensor(softmax_pred),torch.Tensor(true)))
    softmax_pred = torch.Tensor(softmax_pred).argmax(dim=-1)
    softmax_pred = softmax_pred.numpy()
    print('MAE:',masked_mae(true,softmax_pred))
    print('MAPE:',masked_mape(true,softmax_pred))
    print('RMSE:',masked_rmse(true,softmax_pred))
    
    print('-'*10,'Original')
    print('Accuracy:',accuracy_numerical(torch.Tensor(orign_pred.astype(int)),torch.Tensor(true)))
    print('MAE:',masked_mae(true,orign_pred))
    print('MAPE:',masked_mape(true,orign_pred))
    print('RMSE:',masked_rmse(true,orign_pred))
    
# def softmax_calibration_error(softmax_pred,true):

def calculate_ece(true, softmax_pred, bin_num=10):
    """
    Function to calculate Expected Calibration Error (ECE)
    """
    # Get confidence (max softmax output across classes) and predicted class
    confidences = np.max(softmax_pred, axis=-1).flatten()
    predictions = np.argmax(softmax_pred, axis=-1).flatten()

    # Bin the confidences
    bins = np.linspace(0, 1, bin_num+1)
    bin_indices = np.digitize(confidences, bins) - 1

    # Calculate accuracy and confidence for each bin
    bin_accuracy = np.empty(bin_num)
    bin_confidence = np.empty(bin_num)
    for i in range(bin_num):
        bin_items = (bin_indices == i)
        if bin_items.any():
            bin_accuracy[i] = np.mean(true.flatten()[bin_items] == predictions[bin_items])
            bin_confidence[i] = np.mean(confidences[bin_items])
        else:
            bin_accuracy[i] = np.nan
            bin_confidence[i] = np.nan

    # Calculate ECE
    ece = np.nanmean(np.abs(bin_accuracy - bin_confidence))

    return ece


def negative_binomial_calibration_error(mu, alpha, true, bin_num=10):
    """
    Function to calculate Uncalibrated Expected Calibration Error (UCE) 
    for negative binomial regression.
    """
    # Flatten inputs
    mu_flat = mu.flatten()
    true_flat = true.flatten()

    # Calculate the variance for each prediction
    var_flat = mu_flat + alpha * np.square(mu_flat)

    # Bin the predicted variances
    bins = np.linspace(np.min(var_flat), np.max(var_flat), bin_num+1)
    bin_indices = np.digitize(var_flat, bins) - 1

    # Calculate true count and predicted count for each bin
    bin_true_count = np.empty(bin_num)
    bin_predicted_count = np.empty(bin_num)
    bin_true_variance = np.empty(bin_num)
    bin_predicted_variance = np.empty(bin_num)
    for i in range(bin_num):
        bin_items = (bin_indices == i)
        if bin_items.any():
            bin_true_count[i] = np.mean(true_flat[bin_items])
            bin_predicted_count[i] = np.mean(mu_flat[bin_items])
            bin_true_variance[i] = np.var(true_flat[bin_items])
            bin_predicted_variance[i] = np.mean(var_flat[bin_items])
        else:
            bin_true_count[i] = np.nan
            bin_predicted_count[i] = np.nan
            bin_true_variance[i] = np.nan
            bin_predicted_variance[i] = np.nan

    # Calculate UCE
    uce = np.nansum(np.abs(bin_true_count - bin_predicted_count) + 
                    np.abs(bin_true_variance - bin_predicted_variance))

    return uce
def calculate_uce(pred, variance, true, bin_num=10):
    """
    Function to calculate Uncalibrated Expected Calibration Error (UCE)
    """
    # Flatten inputs
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    variance_flat = variance.flatten()

    # Bin the predicted variances
    bins = np.linspace(np.min(variance_flat), np.max(variance_flat), bin_num+1)
    bin_indices = np.digitize(variance_flat, bins) - 1

    # Calculate true and predicted statistics for each bin
    bin_true_mean = np.empty(bin_num)
    bin_pred_mean = np.empty(bin_num)
    bin_true_variance = np.empty(bin_num)
    bin_pred_variance = np.empty(bin_num)
    
    bin_err = np.empty(bin_num)
    bin_uncert = np.empty(bin_num)
    
    for i in range(bin_num):
        bin_items = (bin_indices == i)
        if bin_items.any():
            bin_true_mean[i] = np.mean(true_flat[bin_items])
            bin_pred_mean[i] = np.mean(pred_flat[bin_items])
            bin_err[i] = np.mean(np.square(bin_true_mean[i] - bin_pred_mean[i]))
            
            bin_true_variance[i] = np.var(true_flat[bin_items])
            bin_pred_variance[i] = np.mean(variance_flat[bin_items])
        else:
            bin_true_mean[i] = np.nan
            bin_pred_mean[i] = np.nan
            bin_true_variance[i] = np.nan
            bin_pred_variance[i] = np.nan

    # Calculate UCE
    # errs = np.mean(np.square(bin_true_mean - bin_pred_mean))
    errs    = bin_err
    uncert = bin_pred_variance
    # print(np.nanmin(uncert),bin_true_variance,bin_pred_variance)
    assert np.nanmin(uncert) >= 0 or np.isnan(np.nanmin(uncert))
    # uce = np.nansum( np.abs(bin_true_mean - bin_pred_mean) + np.abs(bin_true_variance - bin_pred_variance))
    uce = np.nansum( np.abs(errs - uncert))
    var_diff = np.nansum(np.abs(bin_true_variance - bin_pred_variance))

    return uce/bin_num,var_diff/bin_num


def calculate_ence(pred, variance, true, bin_num=10):
    """
    Function to calculate Expected Normalized Calibration Error (ENCE)
    """
    # Flatten inputs
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    variance_flat = variance.flatten()**2
    c = 0.303

    # Bin the predicted variances
    bins = np.linspace(np.min(variance_flat), np.max(variance_flat), bin_num+1)
    bin_indices = np.digitize(variance_flat, bins) - 1

    # Calculate true and predicted statistics for each bin
    bin_rmse = np.empty(bin_num)
    bin_rmv = np.empty(bin_num)
    for i in range(bin_num):
        bin_items = (bin_indices == i)
        if bin_items.any():
            residuals = true_flat[bin_items] - pred_flat[bin_items]
            bin_rmse[i] = np.sqrt(np.mean(residuals**2))
            bin_rmv[i] = np.sqrt(np.mean(variance_flat[bin_items]))
        elif (bin_indices == i+1).any():
            bin_items = (bin_indices == i+1)
            residuals = true_flat[bin_items] - pred_flat[bin_items]
            bin_rmse[i] = np.sqrt(np.mean(residuals**2))
            bin_rmv[i] = np.mean(variance_flat[bin_items])
        else:
            bin_rmse[i] = np.nan
            bin_rmv[i] = np.nan

    # Calculate ENCE
    ence = np.nansum(np.abs(bin_rmse - c*bin_rmv) / (c*bin_rmv)) / bin_num
    return ence

def calculate_PICP(upper, lower, true):
    """
    Function to calculate Probability Interval Coverage Probability (PICP).
    
    Args:
        upper: array-like of upper bounds of prediction intervals.
        lower: array-like of lower bounds of prediction intervals.
        true: array-like of true target values.
        
    Returns:
        picp: float, the PICP value.
    """
    in_interval = np.logical_and(true >= lower, true <= upper)
    picp = np.mean(in_interval)
    
    return picp

def calculate_MPIW(upper, lower):
    """
    Function to calculate Mean Predictive Interval Width (MPIW).
    
    Args:
        upper: array-like of upper bounds of prediction intervals.
        lower: array-like of lower bounds of prediction intervals.
        
    Returns:
        mpiw: float, the MPIW value.
    """
    interval_width = upper - lower
    mpiw = np.mean(interval_width)
    
    return mpiw

def mae(true, pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(true - pred))

def mse(true, pred):
    """
    Mean Squared Error
    """
    return np.mean((true - pred)**2)

def rmse(true, pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse(true, pred))

def mape(true, pred, epsilon=1e-10):
    """
    Mean Absolute Percentage Error
    """
    diff = np.abs((true - pred) / np.clip(np.abs(true), epsilon, None))
    return 100. * np.mean(diff)


def compute_crps(mu, alpha, x):
    # Convert from mu and alpha to the parameters for nbinom
    n = 1 / alpha
    p = n / (n + mu)

    # Define the CDF of the predicted distribution
    def F(y):
        return nbinom.cdf(y, n, p)

    # Define the integrand for the CRPS
    def integrand(y):
        return (F(y) - (y >= x)) ** 2

    # Integrate to get the CRPS
    return np.sqrt(quad(integrand, -np.inf, np.inf)[0])
