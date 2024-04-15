We perform the sensitivity analysis of threshold and the choices on bins as well as the run time discussion.

### Sensitivity analysis of number of bins

We choose the bin number from the range `bin_nums = [1, 5, 10, 15, 20, 25, 30]` and discuss the ENCE change with respect to the bin number. Note that the we fix the threshold to 0.5 when perform the uncertainty calibration. We conduct the sensitivity analysis using the four cases in crime dataset using STGCN-NB model outputs.

Table 1: ENCE on the full observations with respect to different bin numbers.

| dataset/bin number  | 1  | 5 | 10  | 15  | 20 | 25 | 
|---|---|---|---|---|---|---|
|  CCR_1h | 0.513  |  0.302 | 0.299  | 0.198  | 0.249  |  0.292 |   
|  CCR_8h | 0.096  | 0.113  |  0.185 | 0.336  | 0.123  |  0.115 |   
|  CCR_1d | 0.093  | 0.167 | 0.185  | 0.192  |  0.189 | 0.207  |   
|  CCR_1w | 0.074  | 0.216  | 0.323  | 0.242  | 0.254  | 0.272  |   

---

Table 2: ENCE on the zero-only observations with respect to different bin numbers.
| dataset/bin number  | 1  | 5 | 10  | 15  | 20 | 25 |  
|---|---|---|---|---|---|---|
|  CCR_1h | 0.650  |  0.689 |  0.604 | 0.267 |  0.229 | 0.501  |
|  CCR_8h | 0.642  |  0.583 | 0.504  | 0.127 |  0.172 | 0.345  |   
|  CCR_1d | 0.641  |  1.181 |  1.096 | 0.738 |  1.025 |  0.958 |   
|  CCR_1w | 1.445  |  0.950 |  0.361 | 0.113 |  0.363 | 0.567  |   

---

Table 3: Run time of the calibration models on the full observations with respect to different bin numbers.

| dataset/bin number  | 1  | 5 | 10  | 15  | 20 | 25 |  
|---|---|---|---|---|---|---|
|  CCR_1h | 19m52s  |  22m10s |  19m41s | 20m52s  | 18m42s  |  18m04s |   
|  CCR_8h | 6m24s  | 5m33s  |  5m33s | 5m33s  |  5m32s | 5m07s  |   
|  CCR_1d | 2m29s  | 1m54s  |  1m49s | 1m40s  |  1m50s |  1m33s |   
|  CCR_1w | 30s  | 13s  | 14s  | 12s  | 14s  | 10s  |   

# Sensitivity Discussion
From Table 1 and 2, we can find that the bins have the impacts. Both smaller bin size and larger bin numbers might not have a good ENCE in the end. 

It is intuitve to understand. For example, if there is only one bin, all the data points fall within the same bin, thus we capare the average data variability with the average MPIW using all the data. It is hard to differentiate different levels of difference. On the other hand, if only few data points lie witin the bin, the results will also be fluctuating due to few data samples. Therefore, bin number between 10 and 20 is preferrable.

# Runtime discussion
The complexity of the algorithm proportionally depends on the size of the data. From the runtime table, you could see that by reducing the resolution from 1hour to 8 hours, 1day, and 1 week, the runtime drops nearly proportionally to the resolution change.
