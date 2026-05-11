@echo off
REM ============================================
REM Weather dataset experiments
REM FTMamba uses small batch (21 vars × batch = heavy)
REM Baselines use larger batch
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
set ENC_IN=21
set DEC_IN=21
set C_OUT=21

echo ==========================================
echo  Part 1: FTMamba on Weather (batch=8)
echo ==========================================

for %%P in (96 192 336 720) do (
    echo [FTMamba] Weather pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model FTMamba --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in %ENC_IN% --dec_in %DEC_IN% --c_out %C_OUT% --d_model %D_MODEL% --d_ff %D_FF% --d_conv %D_CONV% --expand %EXPAND% --batch_size 8 --dropout %DROPOUT% --train_epochs %EPOCHS% --use_amp --des FTMamba --itr %ITR%
)

echo.
echo ==========================================
echo  Part 2: Baselines on Weather (batch=16)
echo ==========================================

for %%P in (96 192 336 720) do (
    echo [PatchTST] Weather pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model PatchTST --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in %ENC_IN% --dec_in %DEC_IN% --c_out %C_OUT% --d_model %D_MODEL% --d_ff %D_FF% --batch_size 16 --dropout %DROPOUT% --train_epochs %EPOCHS% --des PatchTST --itr %ITR%

    echo [iTransformer] Weather pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model iTransformer --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in %ENC_IN% --dec_in %DEC_IN% --c_out %C_OUT% --d_model %D_MODEL% --d_ff %D_FF% --batch_size 16 --dropout %DROPOUT% --train_epochs %EPOCHS% --des iTransformer --itr %ITR%

    echo [DLinear] Weather pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model DLinear --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --enc_in %ENC_IN% --dec_in %DEC_IN% --c_out %C_OUT% --batch_size 16 --train_epochs %EPOCHS% --des DLinear --itr %ITR%

    echo [TimesNet] Weather pred_len=%%P
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id weather_96_%%P --model TimesNet --data custom --features M --seq_len %SEQ_LEN% --label_len %LABEL_LEN% --pred_len %%P --e_layers %E_LAYERS% --d_layers %D_LAYERS% --enc_in %ENC_IN% --dec_in %DEC_IN% --c_out %C_OUT% --d_model %D_MODEL% --d_ff %D_FF% --batch_size 16 --dropout %DROPOUT% --train_epochs %EPOCHS% --top_k 5 --des TimesNet --itr %ITR%
)

echo.
echo ==========================================
echo  Weather experiments completed!
echo ==========================================
pause
