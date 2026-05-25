@echo off
setlocal enabledelayedexpansion

REM ============================================
REM Reviewer Experiment Suite for Symmetry (Windows)
REM ============================================
REM Usage: Double-click or run in cmd
REM ============================================

set CUDA_VISIBLE_DEVICES=0
set SEQ_LEN=96
set LABEL_LEN=48
set E_LAYERS=3
set D_LAYERS=1
set D_MODEL=512
set D_FF=64
set D_CONV=4
set EXPAND=2
set DROPOUT=0.1
set N_HEADS=8
set ITR=1
set BATCH_SIZE=64

echo ==========================================
echo  Reviewer Experiment Suite for Symmetry
echo ==========================================

REM ============================================================
REM 1. Additional baselines (FEDformer, FreTS, S-Mamba, TimeMachine)
REM ============================================================
echo.
echo === Phase 1: Additional Baselines ===

for %%d in (ETTh1 ETTh2 ETTm1) do (
  for %%p in (96 192 336 720) do (

    echo FEDformer on %%d (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model FEDformer --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% --n_heads %N_HEADS% --version Fourier --mode_select random --modes 64

    echo FreTS on %%d (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model FreTS --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers 2 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% --channel_independence 1

    echo S_Mamba on %%d (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model S_Mamba --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 64 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR%

    echo TimeMachine on %%d (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model TimeMachine --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% --n_heads %N_HEADS%
  )
)

REM Weather (21 variates, smaller batch size)
for %%p in (96 192 336 720) do (
  echo FEDformer on Weather (%%p)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model FEDformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR% --n_heads %N_HEADS% --version Fourier --mode_select random --modes 64

  echo FreTS on Weather (%%p)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model FreTS --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers 2 --d_layers 1 --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR% --channel_independence 1

  echo S_Mamba on Weather (%%p)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model S_Mamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 64 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR%

  echo TimeMachine on Weather (%%p)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model TimeMachine --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR% --n_heads %N_HEADS%
)

echo Phase 1 complete.

REM ============================================================
REM 2. Additional datasets (Electricity, Traffic)
REM ============================================================
echo.
echo === Phase 2: Additional Datasets ===

REM Electricity
for %%p in (96 192 336 720) do (
  for %%m in (FTMamba PatchTST iTransformer Mamba DLinear TimesNet Transformer) do (
    set "EXTRA="
    if "%%m"=="PatchTST" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="iTransformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="Transformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="TimesNet" set "EXTRA=--top_k 5 --num_kernels 6 --n_heads %N_HEADS%"

    echo %%m on Electricity (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 321 --dec_in 321 --c_out 321 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 16 --des Dataset_Exp --itr %ITR% !EXTRA!
  )
)

REM Traffic
for %%p in (96 192 336 720) do (
  for %%m in (FTMamba PatchTST iTransformer Mamba DLinear TimesNet Transformer) do (
    set "EXTRA="
    if "%%m"=="PatchTST" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="iTransformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="Transformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="TimesNet" set "EXTRA=--top_k 5 --num_kernels 6 --n_heads %N_HEADS%"

    echo %%m on Traffic (%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 862 --dec_in 862 --c_out 862 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 8 --des Dataset_Exp --itr %ITR% !EXTRA!
  )
)

echo Phase 2 complete.

REM ============================================================
REM 3. Extended ablation study
REM ============================================================
echo.
echo === Phase 3: Extended Ablation ===

for %%d in (ETTh1 ETTh2 ETTm1) do (
  for %%m in (full no_freq add_fusion pure_mamba freq_only scalar_gate channel_gate patch_gate) do (
    echo %%m on %%d (T=96)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%m_96 --model FTMamba --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size %BATCH_SIZE% --ablation_mode %%m --des Ablation_Exp --itr 1
  )
)

REM Weather ablation
for %%m in (full no_freq add_fusion pure_mamba freq_only scalar_gate channel_gate patch_gate) do (
  echo %%m on Weather (T=96)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%m_96 --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 16 --ablation_mode %%m --des Ablation_Exp --itr 1
)

REM Horizon-stratified ablation on ETTh1
for %%p in (96 192 336 720) do (
  for %%m in (full no_freq add_fusion pure_mamba) do (
    echo %%m on ETTh1 (T=%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%m_%%p --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size %BATCH_SIZE% --ablation_mode %%m --des Ablation_Exp --itr 1
  )
)

echo Phase 3 complete.

REM ============================================================
REM 4. Multi-seed PatchTST (3 seeds)
REM ============================================================
echo.
echo === Phase 4: Multi-seed Baselines ===

for %%d in (ETTh1 ETTh2 ETTm1) do (
  for %%p in (96 192 336 720) do (
    echo PatchTST on %%d (%%p, itr=3)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model PatchTST --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Multiseed_Exp --itr 3 --n_heads %N_HEADS%
  )
)
for %%p in (96 192 336 720) do (
  echo PatchTST on Weather (%%p, itr=3)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model PatchTST --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Multiseed_Exp --itr 3 --n_heads %N_HEADS%
)

echo Phase 4 complete.

REM ============================================================
REM 5. Controlled batch size on Weather
REM ============================================================
echo.
echo === Phase 5: Controlled Batch Size (Weather, bs=16) ===

for %%m in (FTMamba PatchTST iTransformer Mamba DLinear TimesNet Transformer) do (
  for %%p in (96 192 336 720) do (
    set "EXTRA="
    if "%%m"=="PatchTST" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="iTransformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="Transformer" set "EXTRA=--n_heads %N_HEADS%"
    if "%%m"=="TimesNet" set "EXTRA=--top_k 5 --num_kernels 6 --n_heads %N_HEADS%"

    echo %%m on Weather (%%p, bs=16)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_bs16_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 16 --des WeatherBS_Exp --itr %ITR% !EXTRA!
  )
)

echo Phase 5 complete.

echo.
echo ==========================================
echo  All reviewer experiments completed!
echo ==========================================
echo Run: python collect_results.py ^> results_reviewer.txt
pause
