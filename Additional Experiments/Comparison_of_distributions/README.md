We compare the implementation results of Poisson distributin modification. Similar to the 

We still use STGCN as the base model. 

Table 1: RMSE comparison of original model, NB modification, and Poisson modification.

| dataset  | Original STGCN  | STGCN-NB | STGCN-Poisson | 
|---|---|---|---|
|  CCR_1h | 0.637  |  0.703 | 0.581 |
|  CCR_8h | 2.386  |  2.326 | 2.419 |
|  CTC_1h | 0.214  | 0.220  | 0.282 |
|  CTC_8h | 0.633  | 0.726  | 0.962 |  

It is clear that Poisson modification has similar performance with NB distribution based on the experiment results. NB distribution is slightly better. The experiment shows that NB could be a potentially flexible distrbution to capture the uncertainty.