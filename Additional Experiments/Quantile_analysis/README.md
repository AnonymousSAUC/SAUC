We show the comparison of coverage of quantiles before and after the calibration. The coverage is calculated as how many points fall within the confidence interval, following the same notation schema from the manuscript, it is written as:

$Coverage_{0.05-0.95}  = \frac{1}{|U|} \sum_{i \in U} 1_{(y_i \geq \hat{F}_i^{-1}(0.05) \& (y_i \leq \hat{F}_i^{-1}(0.95)) )} \approx 0.9 $

where $\hat{F}_i^{-1}(0.05)$ and $\hat{F}_i^{-1}(0.95)$ are the predicted confidence intervals, which is obtained from the model output or the calibrated model output. We expect the portion true data from the test set that fall within the 5%-95% confidence should be approximated to 0.9.

To have the monotonic comparison, we use the metrics:
$C^* = |0.9-Coverage_{0.05-0.95}|$ to measure the closeness between the predicted confidence intervals and the expected 90% data coverage. This is the absolute values of the subtraction of 0.9 to the coverage of the predicted distribution. Smaller the better, i.e., the ideal case is 0.

We compare the metric $C^*$ using the confidence interval outputed by STGCN-NB before and after our calibration methods.

| dataset  | before calibration  | after calibration | 
|---|---|---|
|  CCR_1h | 0.311  |  0.099 | 
|  CCR_8h | 0.347  |  0.040 | 
|  CCR_1d | 0.364  |  0.056 | 
|  CCR_1w | 0.457  |  0.041 | 
|  CTC_1h | 0.311  |  0.122 | 
|  CTC_8h | 0.313  |  0.128 | 
|  CTC_1d | 0.316  |  0.129 | 
|  CTC_1w | 0.463  |  0.140 | 