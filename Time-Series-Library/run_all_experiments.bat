@echo off
REM ============================================
REM FTMamba Experiment Suite (Windows)
REM ============================================

set CUDA_VISIBLE_DEVICES=0
set SEQ_LEN=96
set LABEL_LEN=48
set E_LAYERS=2
set D_LAYERS=1
set D_MODEL=128
set D_FF=16
set D_CONV=4
set EXPAND=2
set DROPOUT=0.1
set N_HEADS=8
set FACTOR=3
set ITR=1

echo ==========================================
echo  FTMamba Experiment Suite
echo ==========================================

REM ------ ETTh1 ------
echo.
echo [1/4] ETTh1
echo.

for %%P in (96 192 336 720) do (
    echo --- FTMamba ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- PatchTST ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model PatchTST --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- iTransformer ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model iTransformer --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Mamba ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- DLinear ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model DLinear --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --enc_in 7 --dec_in 7 --c_out 7 --des FTMamba_Exp --itr %ITR%

    echo --- TimesNet ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model TimesNet --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --top_k 5 --num_kernels 6 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Transformer ETTh1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_%%P_%%P --model Transformer --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
)

REM ------ ETTh2 ------
echo.
echo [2/4] ETTh2
echo.

for %%P in (96 192 336 720) do (
    echo --- FTMamba ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model FTMamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- PatchTST ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model PatchTST --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- iTransformer ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model iTransformer --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Mamba ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model Mamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- DLinear ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model DLinear --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --enc_in 7 --dec_in 7 --c_out 7 --des FTMamba_Exp --itr %ITR%

    echo --- TimesNet ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model TimesNet --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --top_k 5 --num_kernels 6 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Transformer ETTh2 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%%P_%%P --model Transformer --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
)

REM ------ ETTm1 ------
echo.
echo [3/4] ETTm1
echo.

for %%P in (96 192 336 720) do (
    echo --- FTMamba ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- PatchTST ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model PatchTST --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- iTransformer ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model iTransformer --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Mamba ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model Mamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- DLinear ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model DLinear --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --enc_in 7 --dec_in 7 --c_out 7 --des FTMamba_Exp --itr %ITR%

    echo --- TimesNet ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model TimesNet --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --top_k 5 --num_kernels 6 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Transformer ETTm1 pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_%%P_%%P --model Transformer --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
)

REM ------ Weather ------
echo.
echo [4/4] Weather
echo.

for %%P in (96 192 336 720) do (
    echo --- FTMamba Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- PatchTST Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model PatchTST --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- iTransformer Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model iTransformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Mamba Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model Mamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- DLinear Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model DLinear --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --enc_in 21 --dec_in 21 --c_out 21 --des FTMamba_Exp --itr %ITR%

    echo --- TimesNet Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model TimesNet --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --top_k 5 --num_kernels 6 --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%

    echo --- Transformer Weather pred_len=%%P ---
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_%%P_%%P --model Transformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 21 --dec_in 21 --c_out 21 --d_model %D_MODEL% --d_ff %D_FF% --n_heads %N_HEADS% --factor %FACTOR% --dropout %DROPOUT% --des FTMamba_Exp --itr %ITR%
)

echo.
echo ==========================================
echo  All experiments completed!
echo ==========================================
pause
