#!/bin/bash
# ============================================
# Fix NaN: re-run all NaN experiments with properly compiled Mamba
# ============================================

export CUDA_VISIBLE_DEVICES=0
SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=3
D_LAYERS=1
D_MODEL=512
D_FF=64
D_CONV=4
EXPAND=2
DROPOUT=0.1
EPOCHS=3
ITR=1

echo "=========================================="
echo " Part 1: FTMamba on Weather (old NaN)"
echo "=========================================="

for P in 96 192 336 720; do
    echo "[Weather] pred_len=$P"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/weather/ --data_path weather.csv \
        --model_id weather_96_${P}_fix --model FTMamba --data custom \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 21 --dec_in 21 --c_out 21 \
        --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
        --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --des fix_nan --itr $ITR
done

echo ""
echo "=========================================="
echo " Part 2: ETTh2 seed2021 T=720"
echo "=========================================="

echo "[ETTh2] pred_len=720, seed=2021"
python3 -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
    --model_id ETTh2_96_720_seed2021_fix --model FTMamba --data ETTh2 \
    --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 720 \
    --e_layers $E_LAYERS --d_layers $D_LAYERS \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
    --batch_size 32 --dropout $DROPOUT --train_epochs $EPOCHS \
    --use_amp --fix_seed 2021 --des seed2021_fix --itr $ITR

echo ""
echo "=========================================="
echo " Part 3: ETTm1 multi-seed (seed2021)"
echo "=========================================="

# seed2021: T=96, 336, 720 missing
for P in 96 336 720; do
    echo "[ETTm1] pred_len=$P, seed=2021"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
        --model_id ETTm1_96_${P}_seed2021_fix --model FTMamba --data ETTm1 \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
        --batch_size 32 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --fix_seed 2021 --des seed2021_fix --itr $ITR
done

echo ""
echo "=========================================="
echo " Part 4: ETTm1 multi-seed (seed42)"
echo "=========================================="

# seed42: T=336, 720 missing
for P in 336 720; do
    echo "[ETTm1] pred_len=$P, seed=42"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
        --model_id ETTm1_96_${P}_seed42_fix --model FTMamba --data ETTm1 \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
        --batch_size 32 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --fix_seed 42 --des seed42_fix --itr $ITR
done

echo ""
echo "=========================================="
echo " Part 5: ETTm1 multi-seed (seed1234)"
echo "=========================================="

# seed1234: all 4 missing
for P in 96 192 336 720; do
    echo "[ETTm1] pred_len=$P, seed=1234"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
        --model_id ETTm1_96_${P}_seed1234_fix --model FTMamba --data ETTm1 \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
        --batch_size 32 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --fix_seed 1234 --des seed1234_fix --itr $ITR
done

echo ""
echo "=========================================="
echo " Done! Total: 4+1+3+2+4 = 14 runs"
echo "=========================================="
