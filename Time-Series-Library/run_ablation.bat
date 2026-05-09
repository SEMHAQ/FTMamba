@echo off
REM ============================================
REM Ablation study: ETTh1, pred_len=96 only
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
echo  Ablation Study on ETTh1 (pred_len=96)
echo ==========================================

echo.
echo [1] FTMamba (Full model - baseline)
echo.
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96_FTMamba_full --model FTMamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des Ablation_full --itr %ITR%

echo.
echo [2] Mamba only (no frequency branch) - run standard Mamba model
echo.
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96_Mamba_ablation --model Mamba --data ETTh1 --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len 96 --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in 7 --dec_in 7 --c_out 7 --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size %BATCH_SIZE% --dropout %DROPOUT% --use_amp --des Ablation_mamba_only --itr %ITR%

echo.
echo ==========================================
echo  Ablation experiments completed!
echo ==========================================
pause
