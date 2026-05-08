@echo off
REM ============================================
REM Run missing experiments only
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
set FACTOR=3
set ITR=1
set BATCH_SIZE=64

echo ==========================================
echo  Running Missing Experiments
echo ==========================================

REM ------ Weather: FTMamba (reduced batch_size for 21 variables) ------
echo.
echo [1] FTMamba on Weather
echo.

for %%P in (96 192 336 720) do (
    echo --- FTMamba Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 32 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
    if errorlevel 1 echo [WARN] FTMamba Weather pred_len=%%P failed, continuing...
)

REM ------ Missing baselines on all datasets ------
echo.
echo [2] PatchTST on all datasets
echo.

for %%D in (ETTh1 ETTh2 ETTm1) do (
    for %%P in (96 192 336 720) do (
        echo --- PatchTST %%D pred_len=%%P ---
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path %%D.csv --model_id %%D_%%P_%%P --model PatchTST --data %%D --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
        if errorlevel 1 echo [WARN] PatchTST %%D pred_len=%%P failed, continuing...
    )
)

for %%P in (96 192 336 720) do (
    echo --- PatchTST Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model PatchTST --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size 32 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
    if errorlevel 1 echo [WARN] PatchTST Weather pred_len=%%P failed, continuing...
)

echo.
echo [3] iTransformer on all datasets
echo.

for %%D in (ETTh1 ETTh2 ETTm1) do (
    for %%P in (96 192 336 720) do (
        echo --- iTransformer %%D pred_len=%%P ---
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path %%D.csv --model_id %%D_%%P_%%P --model iTransformer --data %%D --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
        if errorlevel 1 echo [WARN] iTransformer %%D pred_len=%%P failed, continuing...
    )
)

for %%P in (96 192 336 720) do (
    echo --- iTransformer Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model iTransformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size 32 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
    if errorlevel 1 echo [WARN] iTransformer Weather pred_len=%%P failed, continuing...
)

echo.
echo [4] Mamba on all datasets
echo.

for %%D in (ETTh1 ETTh2 ETTm1) do (
    for %%P in (96 192 336 720) do (
        echo --- Mamba %%D pred_len=%%P ---
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path %%D.csv --model_id %%D_%%P_%%P --model Mamba --data %%D --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
        if errorlevel 1 echo [WARN] Mamba %%D pred_len=%%P failed, continuing...
    )
)

for %%P in (96 192 336 720) do (
    echo --- Mamba Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model Mamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 32 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
    if errorlevel 1 echo [WARN] Mamba Weather pred_len=%%P failed, continuing...
)

echo.
echo [5] Transformer on all datasets
echo.

for %%D in (ETTh1 ETTh2 ETTm1) do (
    for %%P in (96 192 336 720) do (
        echo --- Transformer %%D pred_len=%%P ---
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path %%D.csv --model_id %%D_%%P_%%P --model Transformer --data %%D --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
        if errorlevel 1 echo [WARN] Transformer %%D pred_len=%%P failed, continuing...
    )
)

for %%P in (96 192 336 720) do (
    echo --- Transformer Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model Transformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --batch_size 32 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
    if errorlevel 1 echo [WARN] Transformer Weather pred_len=%%P failed, continuing...
)

echo.
echo ==========================================
echo  Missing experiments completed!
echo ==========================================
pause
