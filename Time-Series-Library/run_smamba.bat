@echo off
REM ============================================
REM S-Mamba baseline experiments
REM 3 datasets x 4 horizons = 12 runs
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
echo  S-Mamba Baseline Experiments
echo ==========================================

REM === ETTh1 ===
echo.
echo [1/12] ETTh1 pred_len=96
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model S_Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [2/12] ETTh1 pred_len=192
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_192 --model S_Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [3/12] ETTh1 pred_len=336
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_336 --model S_Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [4/12] ETTh1 pred_len=720
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_720 --model S_Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

REM === ETTh2 ===
echo [5/12] ETTh2 pred_len=96
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_96 --model S_Mamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [6/12] ETTh2 pred_len=192
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_192 --model S_Mamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [7/12] ETTh2 pred_len=336
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_336 --model S_Mamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [8/12] ETTh2 pred_len=720
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_720 --model S_Mamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

REM === ETTm1 ===
echo [9/12] ETTm1 pred_len=96
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_96 --model S_Mamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [10/12] ETTm1 pred_len=192
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_192 --model S_Mamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [11/12] ETTm1 pred_len=336
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_336 --model S_Mamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo [12/12] ETTm1 pred_len=720
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_720 --model S_Mamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des S_Mamba --itr %ITR%

echo.
echo ==========================================
echo  S-Mamba experiments completed!
echo ==========================================
pause
