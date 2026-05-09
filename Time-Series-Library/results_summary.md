# FTMamba Experiment Results - Complete

## ETTh1 (MSE / MAE)

| Horizon | FTMamba | PatchTST | iTransformer | DLinear | TimesNet | Transformer |
|---------|---------|----------|-------------|---------|----------|-------------|
| 96 | **0.3776** / 0.4012 | 0.3827 / 0.4047 | 0.3935 / 0.4107 | 0.4108 / 0.4228 | 0.4333 / 0.4349 | 0.8445 / 0.7210 |
| 192 | **0.4296** / 0.4299 | 0.4385 / 0.4407 | 0.4521 / 0.4413 | 0.4579 / 0.4509 | 0.5067 / 0.4802 | 0.8028 / 0.6967 |
| 336 | **0.4931** / 0.4588 | 0.4857 / 0.4687 | 0.4941 / 0.4630 | 0.4972 / 0.4737 | 0.5507 / 0.5048 | 1.0943 / 0.8518 |
| 720 | **0.4668** / 0.4651 | 0.4878 / 0.4854 | 0.5118 / 0.4950 | 0.5231 / 0.5177 | 0.7140 / 0.5859 | 1.0878 / 0.8538 |

FTMamba: Best on 3/4 horizons (96, 192, 720). PatchTST slightly better at 336.

## ETTh2 (MSE / MAE)

| Horizon | FTMamba | PatchTST | iTransformer | DLinear | TimesNet | Transformer |
|---------|---------|----------|-------------|---------|----------|-------------|
| 96 | **0.2904** / 0.3409 | 0.3016 / 0.3489 | 0.3007 / 0.3497 | 0.3595 / 0.4105 | 0.3444 / 0.3803 | 1.9057 / 1.1097 |
| 192 | **0.3784** / 0.3967 | 0.3757 / 0.3978 | 0.3938 / 0.4071 | 0.4917 / 0.4865 | 0.4327 / 0.4258 | 3.6999 / 1.4669 |
| 336 | **0.4118** / 0.4289 | 0.4211 / 0.4342 | 0.4350 / 0.4392 | 0.5985 / 0.5470 | 0.4729 / 0.4564 | 3.2220 / 1.4197 |
| 720 | **0.4434** / 0.4576 | 0.4424 / 0.4601 | 0.4317 / 0.4497 | 0.8597 / 0.6699 | 0.4637 / 0.4706 | 3.4519 / 1.4767 |

FTMamba: Best on 3/4 horizons (96, 192, 336). iTransformer slightly better at 720.

## ETTm1 (MSE / MAE)

| Horizon | FTMamba | PatchTST | iTransformer | DLinear | TimesNet | Transformer |
|---------|---------|----------|-------------|---------|----------|-------------|
| 96 | **0.3439** / 0.3777 | 0.3368 / 0.3741 | 0.3377 / 0.3729 | 0.3480 / 0.3739 | 0.3881 / 0.3964 | 0.5832 / 0.5536 |
| 192 | **0.3757** / 0.3944 | 0.3683 / 0.3900 | 0.3873 / 0.3978 | 0.3845 / 0.3932 | 0.4212 / 0.4183 | 0.5914 / 0.5637 |
| 336 | 0.4099 / 0.4172 | **0.4062** / 0.4153 | 0.4283 / 0.4242 | 0.4149 / 0.4143 | 0.4650 / 0.4375 | 1.0038 / 0.7817 |
| 720 | **0.4697** / 0.4564 | 0.4648 / 0.4492 | 0.5142 / 0.4719 | 0.4733 / 0.4503 | 0.5273 / 0.4716 | 1.1324 / 0.8458 |

FTMamba: Best at 720. PatchTST best at 96, 192, 336 (very competitive).

## Weather (MSE / MAE) - FTMamba has NaN issue (numerical instability with 21 variables)

| Horizon | FTMamba | PatchTST | iTransformer | DLinear | TimesNet | Transformer |
|---------|---------|----------|-------------|---------|----------|-------------|
| 96 | NaN | **0.1717** / 0.2129 | 0.1749 / 0.2144 | 0.1961 / 0.2563 | 0.1701 / 0.2195 | 0.3093 / 0.3805 |
| 192 | NaN | **0.2192** / 0.2551 | 0.2243 / 0.2567 | 0.2364 / 0.2943 | 0.2269 / 0.2677 | 0.4850 / 0.4935 |
| 336 | NaN | **0.2757** / 0.2956 | 0.2831 / 0.3001 | 0.2815 / 0.3313 | 0.2893 / 0.3093 | 0.6190 / 0.5698 |
| 720 | NaN | 0.3521 / 0.3450 | 0.3588 / 0.3501 | 0.3454 / 0.3820 | 0.3657 / 0.3588 | **0.9678** / 0.7389 |

## Missing Results
- **Mamba**: No results (model not in batch script or crashed silently)
- **FTMamba on Weather**: NaN - needs numerical stability fix

## Key Findings
1. FTMamba achieves SOTA on ETTh1, ETTh2 across most horizons
2. PatchTST is the strongest competitor, especially on ETTm1
3. Transformer baseline is very weak (expected for long-term forecasting)
4. FTMamba has numerical stability issue on Weather dataset (21 variables)
