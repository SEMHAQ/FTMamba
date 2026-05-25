@echo off
REM ============================================
REM Reviewer Experiment Suite for Symmetry (Windows)
REM ============================================
REM Usage: Double-click or run in cmd
REM Requires: GPU, Python, PyTorch, datasets downloaded
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

for %%d in (ETTh1 ETTh2 ETTm1 Weather) do (
  for %%m in (FEDformer FreTS S_Mamba TimeMachine) do (
    for %%p in (96 192 336 720) do (
      echo Running: %%m on %%d (pred_len=%%p)

      set EXTRA=
      if "%%m"=="FEDformer" set EXTRA=--n_heads %N_HEADS% --version Fourier --mode_select random --modes 64
      if "%%m"=="FreTS" set EXTRA=--channel_independence 1
      if "%%m"=="TimeMachine" set EXTRA=--d_ff 2048 --n_heads %N_HEADS%

      if "%%m"=="FreTS" (
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model %%m --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers 2 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% %EXTRA%
      ) else if "%%m"=="TimeMachine" (
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model %%m --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% %EXTRA%
      ) else (
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model %%m --data %%d --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Review_Exp --itr %ITR% %EXTRA%
      )
      if errorlevel 1 echo Warning: %%m on %%d (%%p) failed
    )
  )
)

REM Special: Weather uses custom data type and 21 enc_in
for %%m in (FEDformer FreTS S_Mamba TimeMachine) do (
  for %%p in (96 192 336 720) do (
    echo Running: %%m on Weather (pred_len=%%p)
    if "%%m"=="FEDformer" (
      python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR% --n_heads %N_HEADS% --version Fourier --mode_select random --modes 64
    ) else if "%%m"=="FreTS" (
      python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers 2 --d_layers 1 --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR% --channel_independence 1
    ) else (
      python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size 16 --des Review_Exp --itr %ITR%
    )
  )
)

echo Phase 1 complete.

REM ============================================================
REM 2. Additional datasets (Electricity, Traffic)
REM ============================================================
echo.
echo === Phase 2: Additional Datasets ===
echo Running Electricity and Traffic...

for %%d in (Electricity Traffic) do (
  if "%%d"=="Electricity" (
    set ENC_IN=321
    set BS=16
    set ROOT=dataset/electricity
    set FILE=electricity.csv
  ) else (
    set ENC_IN=862
    set BS=8
    set ROOT=dataset/traffic
    set FILE=traffic.csv
  )

  for %%m in (FTMamba PatchTST iTransformer Mamba DLinear TimesNet Transformer) do (
    for %%p in (96 192 336 720) do (
      echo Running: %%m on %%d (pred_len=%%p)

      set EXTRA=
      if "%%m"=="PatchTST" set EXTRA=--n_heads %N_HEADS%
      if "%%m"=="iTransformer" set EXTRA=--n_heads %N_HEADS%
      if "%%m"=="Transformer" set EXTRA=--n_heads %N_HEADS%
      if "%%m"=="TimesNet" set EXTRA=--top_k 5 --num_kernels 6 --n_heads %N_HEADS%

      python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./%ROOT%/ --data_path %FILE% --model_id %%d_%%p_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in %ENC_IN% --dec_in %ENC_IN% --c_out %ENC_IN% --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size %BS% --des Dataset_Exp --itr %ITR% %EXTRA%
    )
  )
)
echo Phase 2 complete.

REM ============================================================
REM 3. Extended ablation study
REM ============================================================
echo.
echo === Phase 3: Extended Ablation ===

REM 3a. Multi-dataset ablation (all 4 datasets, T=96, 8 modes)
for %%d in (ETTh1 ETTh2 ETTm1 Weather) do (
  set ABL_BS=%BATCH_SIZE%
  if "%%d"=="Weather" set ABL_BS=16

  for %%m in (full no_freq add_fusion pure_mamba freq_only scalar_gate channel_gate patch_gate) do (
    echo Ablation: %%m on %%d (T=96)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%m_96 --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size %ABL_BS% --ablation_mode %%m --des Ablation_Exp --itr 1
  )
)

REM Weather with 21 enc_in
for %%m in (full no_freq add_fusion pure_mamba freq_only scalar_gate channel_gate patch_gate) do (
  echo Ablation: %%m on Weather (T=96)
  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_%%m_96 --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 16 --ablation_mode %%m --des Ablation_Exp --itr 1
)

REM 3b. Horizon-stratified ablation (ETTh1, 4 horizons, 4 modes)
for %%p in (96 192 336 720) do (
  for %%m in (full no_freq add_fusion pure_mamba) do (
    echo Ablation: %%m on ETTh1 (T=%%p)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%m_%%p --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size %BATCH_SIZE% --ablation_mode %%m --des Ablation_Exp --itr 1
  )
)
echo Phase 3 complete.

REM ============================================================
REM 4. Multi-seed PatchTST (3 seeds)
REM ============================================================
echo.
echo === Phase 4: Multi-seed Baselines ===

for %%d in (ETTh1 ETTh2 ETTm1 Weather) do (
  for %%p in (96 192 336 720) do (
    echo Running: PatchTST on %%d (pred_len=%%p, itr=3)
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/%%d/ --data_path %%d.csv --model_id %%d_%%p_%%p --model PatchTST --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff 2048 --dropout %DROPOUT% --batch_size %BATCH_SIZE% --des Multiseed_Exp --itr 3 --n_heads %N_HEADS%
  )
)
REM Weather (21 enc_in)
for %%p in (96 192 336 720) do (
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
    echo Running: %%m on Weather (pred_len=%%p, bs=16)

    set EXTRA=
    if "%%m"=="PatchTST" set EXTRA=--n_heads %N_HEADS%
    if "%%m"=="iTransformer" set EXTRA=--n_heads %N_HEADS%
    if "%%m"=="Transformer" set EXTRA=--n_heads %N_HEADS%
    if "%%m"=="TimesNet" set EXTRA=--top_k 5 --num_kernels 6 --n_heads %N_HEADS%

    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_bs16_%%p --model %%m --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%p --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --batch_size 16 --des WeatherBS_Exp --itr %ITR% %EXTRA%
  )
)
echo Phase 5 complete.

REM ============================================================
REM 6. Efficiency measurement
REM ============================================================
echo.
echo === Phase 6: Efficiency Metrics ===
python -u run_efficiency.py
echo Phase 6 complete.

REM ============================================================
REM 7. Gate analysis
REM ============================================================
echo.
echo === Phase 7: Gate Analysis ===
python -u run_gate_analysis.py
echo Phase 7 complete.

REM ============================================================
REM 8. Collect results
REM ============================================================
echo.
echo === Phase 8: Collect Results ===
python collect_results.py > results_reviewer.txt
echo Results saved to results_reviewer.txt

echo.
echo ==========================================
echo  All reviewer experiments completed!
echo ==========================================
pause
