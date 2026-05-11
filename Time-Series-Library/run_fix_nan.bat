@echo off
REM ============================================
REM Fix NaN: ETTm1 multi-seed + Weather FTMamba
REM ETTm1: 3 seeds x 4 horizons = 12 runs
REM Weather: 4 horizons = 4 runs
REM Total: 16 runs
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
set BATCH_SIZE=32
set ITR=1

echo ==========================================
echo  Part 1: ETTm1 multi-seed (NaN fix)
echo ==========================================

for %%S in (2021 42 1234) do (
    for %%P in (96 192 336 720) do (
        echo [ETTm1] pred_len=%%P, seed=%%S
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_%%P_seed%%S --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --train_epochs 5 --use_amp --fix_seed %%S --des seed%%S --itr %ITR%
    )
)

echo.
echo ==========================================
echo  Part 2: Weather FTMamba (NaN fix)
echo ==========================================

for %%P in (96 192 336 720) do (
    echo [Weather] pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --train_epochs 5 --use_amp --des FTMamba --itr %ITR%
)

echo.
echo  Done! Total: 16 runs
pause
