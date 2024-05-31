from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np

def calibrate_model(mu, var, y, upper, lower, split_index, bin_num=15):
    # Split data into validation and test sets
    val_mu, test_mu = mu[:split_index], mu[split_index:]
    val_var, test_var = var[:split_index], var[split_index:]
    val_y, test_y = y[:split_index], y[split_index:]
    val_upper, test_upper = upper[:split_index], upper[split_index:]
    val_lower, test_lower = lower[:split_index], lower[split_index:]

    # Flatten the arrays
    val_mu_flat, test_mu_flat = val_mu.flatten(), test_mu.flatten()
    val_var_flat, test_var_flat = val_var.flatten(), test_var.flatten()
    val_y_flat, test_y_flat = val_y.flatten(), test_y.flatten()
    val_upper_flat, test_upper_flat = val_upper.flatten(), test_upper.flatten()
    val_lower_flat, test_lower_flat = val_lower.flatten(), test_lower.flatten()

    # Initialize your bins
    percentiles = np.linspace(0, 100, bin_num+1)
    bins = np.percentile(val_y_flat, percentiles)

    # Initialize empty arrays to hold the calibrated predicted values and variances
    val_calibrated_mu = np.zeros(val_mu_flat.shape)
    val_calibrated_upper = np.zeros(val_var_flat.shape)
    val_calibrated_lower = np.zeros(val_var_flat.shape)
    val_calibrated_var = np.zeros(val_var_flat.shape)

    test_calibrated_mu = np.zeros(test_mu_flat.shape)
    test_calibrated_upper = np.zeros(test_var_flat.shape)
    test_calibrated_lower = np.zeros(test_var_flat.shape)
    test_calibrated_var = np.zeros(test_var_flat.shape)

    up = 0.95
    low = 0.05

    for i in tqdm(range(bin_num)):
        # Get indices for current bin
        bin_indices = (val_y_flat >= bins[i]) & (val_y_flat <= bins[i+1])

        # Separate zero and non-zero indices for validation set
        val_zero_indices = (val_mu_flat<0.5) & bin_indices
        val_non_zero_indices = (val_mu_flat>=0.5) & bin_indices

        # Separate zero and non-zero indices for test set
        test_zero_indices = (test_mu_flat<0.5)
        test_non_zero_indices = (test_mu_flat>=0.5)

        # Perform quantile regression and use it to adjust the predicted values and variances
        if sum(val_non_zero_indices) > 0:
            X_non_zero = sm.add_constant(val_mu_flat[val_non_zero_indices])
            res_non_zero = QuantReg(val_y_flat[val_non_zero_indices], X_non_zero).fit(q=.5)
            upper_non_zero = QuantReg(val_y_flat[val_non_zero_indices], X_non_zero).fit(q=up)
            lower_non_zero = QuantReg(val_y_flat[val_non_zero_indices], X_non_zero).fit(q=low)

            val_calibrated_mu[val_non_zero_indices] = res_non_zero.predict(X_non_zero)
            val_calibrated_upper[val_non_zero_indices] = upper_non_zero.predict(X_non_zero)
            val_calibrated_lower[val_non_zero_indices] = lower_non_zero.predict(X_non_zero)
            val_calibrated_var[val_non_zero_indices] = val_calibrated_upper[val_non_zero_indices] - val_calibrated_lower[val_non_zero_indices]

            # Apply the non-zero model to the test set
            test_X_non_zero = sm.add_constant(test_mu_flat[test_non_zero_indices])
            test_calibrated_mu[test_non_zero_indices] = res_non_zero.predict(test_X_non_zero)
            test_calibrated_upper[test_non_zero_indices] = upper_non_zero.predict(test_X_non_zero)
            test_calibrated_lower[test_non_zero_indices] = lower_non_zero.predict(test_X_non_zero)
            test_calibrated_var[test_non_zero_indices] = test_calibrated_upper[test_non_zero_indices] - test_calibrated_lower[test_non_zero_indices]

        # For zero indices, perform quantile regression if we have enough data
        if sum(val_zero_indices) > 0:
            X_zero = sm.add_constant(val_mu_flat[val_zero_indices])
            res_zero = QuantReg(val_y_flat[val_zero_indices], X_zero).fit(q=.5)
            upper_zero = QuantReg(val_y_flat[val_zero_indices], X_zero).fit(q=up)
            lower_zero = QuantReg(val_y_flat[val_zero_indices], X_zero).fit(q=low)

            val_calibrated_mu[val_zero_indices] = res_zero.predict(X_zero)
            val_calibrated_upper[val_zero_indices] = upper_zero.predict(X_zero)
            val_calibrated_lower[val_zero_indices] = lower_zero.predict(X_zero)
            val_calibrated_var[val_zero_indices] = val_calibrated_upper[val_zero_indices] - val_calibrated_lower[val_zero_indices]

            # Apply the zero model to the test set
            test_X_zero = sm.add_constant(test_mu_flat[test_zero_indices])
            test_calibrated_mu[test_zero_indices] = res_zero.predict(test_X_zero)
            test_calibrated_upper[test_zero_indices] = upper_zero.predict(test_X_zero)
            test_calibrated_lower[test_zero_indices] = lower_zero.predict(test_X_zero)
            test_calibrated_var[test_zero_indices] = test_calibrated_upper[test_zero_indices] - test_calibrated_lower[test_zero_indices]

    return (val_calibrated_mu, val_calibrated_var, val_calibrated_upper, val_calibrated_lower,
            test_calibrated_mu, test_calibrated_var, test_calibrated_upper, test_calibrated_lower,val_y,test_y)
