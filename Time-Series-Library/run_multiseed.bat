@echo off
REM ============================================
REM Multi-seed FTMamba experiments
REM 3 seeds x 3 datasets x 4 horizons = 36 runs
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
echo  Multi-seed FTMamba Experiments
echo ==========================================

for %%S in (2021 42 1234) do (
    echo.
    echo ==========================================
    echo  Seed = %%S
    echo ==========================================

    REM === ETTh1 ===
    echo [ETTh1] pred_len=96, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96_seed%%S --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh1] pred_len=192, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_192_seed%%S --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh1] pred_len=336, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_336_seed%%S --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh1] pred_len=720, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_720_seed%%S --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    REM === ETTh2 ===
    echo [ETTh2] pred_len=96, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_96_seed%%S --model FTMamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh2] pred_len=192, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_192_seed%%S --model FTMamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh2] pred_len=336, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_336_seed%%S --model FTMamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTh2] pred_len=720, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_96_720_seed%%S --model FTMamba --data ETTh2 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    REM === ETTm1 ===
    echo [ETTm1] pred_len=96, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_96_seed%%S --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTm1] pred_len=192, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_192_seed%%S --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 192 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTm1] pred_len=336, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_336_seed%%S --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 336 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%

    echo [ETTm1] pred_len=720, seed=%%S
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1_96_720_seed%%S --model FTMamba --data ETTm1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 720 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --fix_seed %%S --des seed%%S --itr %ITR%
)

echo.
echo ==========================================
echo  Multi-seed experiments completed!
echo ==========================================
pause
