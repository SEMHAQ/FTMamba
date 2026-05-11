@echo off
REM ============================================
REM Fix NaN: remaining runs only
REM ETTm1: batch=32 (7 vars)
REM Weather: batch=8 (21 vars)
REM Skips already-completed seed2021 runs
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
set EPOCHS=3
set ITR=1

echo ==========================================
echo  Part 1: ETTm1 remaining multi-seed
echo ==========================================

REM seed2021: only 720 missing
echo [ETTm1] pred_len=720, seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_720_seed2021 --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 32 --dropout %DROPOUT% --train_epochs %EPOCHS% --use_amp --fix_seed 2021 --des seed2021 --itr %ITR%

REM seed42: 336 and 720 missing
for %%P in (336 720) do (
    echo [ETTm1] pred_len=%%P, seed=42
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_%%P_seed42 --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 32 --dropout %DROPOUT% --train_epochs %EPOCHS% --use_amp --fix_seed 42 --des seed42 --itr %ITR%
)

REM seed1234: all 4 missing
for %%P in (96 192 336 720) do (
    echo [ETTm1] pred_len=%%P, seed=1234
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_%%P_seed1234 --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 32 --dropout %DROPOUT% --train_epochs %EPOCHS% --use_amp --fix_seed 1234 --des seed1234 --itr %ITR%
)

echo.
echo ==========================================
echo  Part 2: Weather FTMamba multi-seed (batch=8)
echo ==========================================

for %%S in (2021 42 1234) do (
    for %%P in (96 192 336 720) do (
        echo [Weather] pred_len=%%P, seed=%%S
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P_seed%%S --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 8 --dropout %DROPOUT% --train_epochs %EPOCHS% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%
    )
)

echo.
echo  Done! Remaining: 7 ETTm1 + 12 Weather = 19 runs
pause
