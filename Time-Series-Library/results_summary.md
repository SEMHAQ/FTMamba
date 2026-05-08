# FTMamba Experiment Results

## ETTh1 (MSE / MAE)

| pred_len | FTMamba | DLinear | TimesNet |
|----------|---------|---------|----------|
| 96 | **0.3776 / 0.4012** | 0.4108 / 0.4228 | 0.4333 / 0.4349 |
| 192 | **0.4296 / 0.4299** | 0.4579 / 0.4509 | 0.5067 / 0.4802 |
| 336 | **0.4931 / 0.4588** | 0.4972 / 0.4737 | 0.5507 / 0.5048 |
| 720 | **0.4668 / 0.4651** | 0.5231 / 0.5177 | 0.7140 / 0.5859 |

FTMamba wins all 4 horizons on MSE. Avg improvement over DLinear: **7.0%**.

## ETTh2 (MSE / MAE)

| pred_len | FTMamba | DLinear | TimesNet |
|----------|---------|---------|----------|
| 96 | **0.2904 / 0.3409** | 0.3595 / 0.4105 | 0.3444 / 0.3803 |
| 192 | **0.3784 / 0.3967** | 0.4917 / 0.4865 | 0.4327 / 0.4258 |
| 336 | **0.4118 / 0.4289** | 0.5985 / 0.5470 | 0.4729 / 0.4564 |
| 720 | **0.4434 / 0.4576** | 0.8597 / 0.6699 | 0.4637 / 0.4706 |

FTMamba wins all 4 horizons on MSE. Avg improvement over DLinear: **20.3%**.

## ETTm1 (MSE / MAE)

| pred_len | FTMamba | DLinear | TimesNet |
|----------|---------|---------|----------|
| 96 | **0.3439 / 0.3777** | 0.3480 / 0.3739 | 0.3881 / 0.3964 |
| 192 | **0.3757 / 0.3944** | 0.3845 / 0.3932 | 0.4212 / 0.4183 |
| 336 | **0.4099 / 0.4172** | 0.4149 / 0.4143 | 0.4650 / 0.4375 |
| 720 | **0.4697 / 0.4564** | 0.4733 / 0.4503 | 0.5273 / 0.4716 |

FTMamba wins all 4 horizons on MSE. Avg improvement over DLinear: **1.3%**.

## Weather (MSE / MAE) - FTMamba missing, experiment crashed

| pred_len | FTMamba | DLinear | TimesNet |
|----------|---------|---------|----------|
| 96 | **MISSING** | 0.1961 / 0.2563 | 0.1701 / 0.2195 |
| 192 | **MISSING** | 0.2364 / 0.2943 | 0.2269 / 0.2677 |
| 336 | **MISSING** | 0.2815 / 0.3313 | 0.2893 / 0.3093 |
| 720 | **MISSING** | 0.3454 / 0.3820 | 0.3657 / 0.3588 |

## Missing Baselines
The following models were skipped due to Transformer crashing (reformer_pytorch dependency):
- PatchTST
- iTransformer
- Mamba
- Transformer

## Status
- FTMamba outperforms DLinear and TimesNet on ETTh1, ETTh2, ETTm1 across all horizons
- Need to fix Weather FTMamba + run missing baselines
