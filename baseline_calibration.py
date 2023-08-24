from scipy.stats import norm
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from scipy.special import expit
import numpy as np

# Define the objective function for minimization
def objective(T, mean, true):
    # Compute negative log likelihood
    scaled_mean = mean / T
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * (true - scaled_mean)**2
    return np.mean(nll)

def calibrate_temperature(mean_val, variance_val, true_val, mean_test, variance_test, true_test):
    # Optimize temperature
    result = minimize(objective, x0=[1.0], args=(mean_val, true_val), method='L-BFGS-B', bounds=[(0.001, 10.0)])
    T = result.x[0]

    # Apply temperature scaling
    calibrated_mean_test = mean_test / T
    calibrated_variance_test = variance_test / T**2
    return calibrated_mean_test, calibrated_variance_test

def calibrate_isotonic(mean_val, variance_val, true_val, mean_test, variance_test, true_test):
    # Initialize Isotonic Regression model
    ir = IsotonicRegression(out_of_bounds='clip')

    # Fit isotonic regression model and calibrate mean
    ir.fit(mean_val, true_val)
    calibrated_mean_test = ir.transform(mean_test)

    # Variance calibration not applicable in isotonic regression
    calibrated_variance_test = variance_test 
    return calibrated_mean_test, calibrated_variance_test

def calibrate_platt(mean_val, variance_val, true_val, mean_test, variance_test, true_test):
    # Initialize Logistic Regression model
    lr = LogisticRegression(solver='liblinear')

    # Fit logistic regression model and calibrate mean
    lr.fit(mean_val[:, None], true_val)
    calibrated_mean_test = lr.predict(mean_test[:, None])

    # Variance calibration not applicable in Platt Scaling
    calibrated_variance_test = variance_test
    return calibrated_mean_test, calibrated_variance_test

def calibrate_histogram(mean_val, variance_val, true_val, mean_test, variance_test, true_test, bin_num=10):
    # Initialize Bins Discretizer
    bd = KBinsDiscretizer(n_bins=bin_num, strategy='uniform', encode='ordinal')

    # Fit bins discretizer and calibrate mean
    bd.fit(mean_val[:, None])
    bin_indices = bd.transform(mean_test[:, None]).astype(int)
    calibrated_mean_test = np.empty_like(mean_test)
    for bin_idx in range(bin_num):
        bin_items = (bin_indices == bin_idx)
        if np.any(bin_items):
            calibrated_mean_test[bin_items.T[0,:]] = true_test[bin_indices.T[0,:]==bin_idx].mean()

    # Variance calibration not applicable in Histogram Binning
    calibrated_variance_test = variance_test
    return calibrated_mean_test, calibrated_variance_test

def calibrate_regression(mean_val, variance_val, true_val, mean_test, variance_test, true_test, method='temperature'):
    if method == 'temperature':
        return calibrate_temperature(mean_val, variance_val, true_val, mean_test, variance_test, true_test)
    elif method == 'isotonic':
        return calibrate_isotonic(mean_val, variance_val, true_val, mean_test, variance_test, true_test)
    elif method == 'platt':
        return calibrate_platt(mean_val, variance_val, true_val, mean_test, variance_test, true_test)
    elif method == 'histogram':
        return calibrate_histogram(mean_val, variance_val, true_val, mean_test, variance_test, true_test)
    else:
        raise ValueError("Invalid method. Must be 'temperature', 'isotonic', 'platt', 'histogram', or 'beta'.")
    
    
def apply_calibration_methods(intervals, crash_nb_mu, crash_nb_true, crash_nb_var, methods=['temperature', 'isotonic', 'platt', 'histogram']):
    
    calib_results = {}

    for method in tqdm(methods, total=len(methods)):
        calib_results[method] = {}
        for interval in intervals:
            stgcn_crash_nb_mu = crash_nb_mu[interval]
            stgcn_crash_true = crash_nb_true[interval]
            stgcn_crash_nb_var = crash_nb_var[interval]
            
            # Split data into validation and test sets
            split_index = int(len(stgcn_crash_nb_mu) * 0.4)
            val_mu, test_mu = stgcn_crash_nb_mu[:split_index], stgcn_crash_nb_mu[split_index:]
            val_true, test_true = stgcn_crash_true[:split_index], stgcn_crash_true[split_index:]
            val_var, test_var = stgcn_crash_nb_var[:split_index], stgcn_crash_nb_var[split_index:]

            # Flatten the data for calibration
            val_mu_flat, test_mu_flat = val_mu.flatten(), test_mu.flatten()
            val_true_flat, test_true_flat = val_true.flatten(), test_true.flatten()
            val_var_flat, test_var_flat = val_var.flatten(), test_var.flatten()
            
            # Apply calibration
            mean_calib, var_calib = calibrate_regression(val_mu_flat, val_var_flat, val_true_flat, 
                                                         test_mu_flat, test_var_flat, test_true_flat, 
                                                         method=method)

            # Reshape the calibrated data back to original shape and assign to the whole data
            calibrated_mu = stgcn_crash_nb_mu.copy()
            calibrated_var = stgcn_crash_nb_var.copy()
            calibrated_mu[split_index:] = mean_calib.reshape(test_mu.shape)
            calibrated_var[split_index:] = var_calib.reshape(test_var.shape)

            calib_results[method][interval] = {'mean': calibrated_mu, 'var': calibrated_var}

    return calib_results
