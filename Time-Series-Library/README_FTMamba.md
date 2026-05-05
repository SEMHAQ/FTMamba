# FTMamba: Frequency-aware Temporal Mamba for Long-term Time Series Forecasting

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_ftmamba.txt
```

### 2. Download Datasets
```bash
python download_data.py
```

### 3. Run Single Experiment (FTMamba on ETTh1, pred_len=96)
```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model FTMamba \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 16 \
  --d_conv 4 \
  --expand 2 \
  --dropout 0.1 \
  --des 'FTMamba_Exp' \
  --itr 1
```

### 4. Run All Experiments (FTMamba + all baselines)
```bash
bash run_all_experiments.sh
```

This runs: FTMamba, PatchTST, iTransformer, Mamba, DLinear, TimesNet, Transformer
on: ETTh1, ETTh2, ETTm1, Weather
with prediction lengths: 96, 192, 336, 720

### 5. Collect Results
```bash
python collect_results.py
```

## Project Structure

```
Time-Series-Library/
├── models/
│   ├── FTMamba.py          # Our model (NEW)
│   ├── PatchTST.py         # Baseline
│   ├── iTransformer.py     # Baseline
│   ├── Mamba.py            # Baseline
│   ├── DLinear.py          # Baseline
│   ├── TimesNet.py         # Baseline
│   └── Transformer.py      # Baseline
├── scripts/long_term_forecast/
│   └── FTMamba_script/     # Our experiment scripts
├── dataset/                 # Datasets (auto-downloaded)
├── run.py                   # Main entry point
├── download_data.py         # Dataset downloader
├── run_all_experiments.sh   # Master experiment script
└── collect_results.py       # Result formatter
```

## Model Architecture

FTMamba uses a dual-branch architecture:

1. **Patch Embedding**: Splits time series into patches (from PatchTST)
2. **Temporal Branch**: Mamba blocks for long-range dependency modeling O(L)
3. **Frequency Branch**: FFT + learnable frequency filter + iFFT
4. **Gated Fusion**: Learnable gate to adaptively combine both branches
5. **Prediction Head**: Flatten head for output

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_len | 96 | Input sequence length |
| pred_len | 96/192/336/720 | Prediction length |
| d_model | 128 | Model dimension |
| d_ff | 16 | SSM state dimension |
| d_conv | 4 | Convolution kernel size |
| expand | 2 | Expansion factor |
| e_layers | 2 | Number of FTMamba layers |
| patch_len | 16 | Patch length (in FTMamba.py) |
| stride | 8 | Patch stride (in FTMamba.py) |
