#!/bin/bash
# ============================================
# Run missing experiments for Electronics paper
# - Electricity T=720
# - Traffic T=192, 336, 720
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
EPOCHS=10
ITR=1

echo "=========================================="
echo " Part 1: Electricity T=720 (all models)"
echo "=========================================="

# FTMamba
echo "[Electricity] FTMamba T=720"
python3 -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id electricity_96_720 --model FTMamba --data custom \
    --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 720 \
    --e_layers $E_LAYERS --d_layers $D_LAYERS \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
    --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
    --use_amp --des Exp --itr $ITR

# PatchTST
echo "[Electricity] PatchTST T=720"
python3 -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id electricity_96_720 --model PatchTST --data custom \
    --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 720 \
    --e_layers $E_LAYERS --d_layers $D_LAYERS \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --d_model $D_MODEL --d_ff $D_FF \
    --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
    --use_amp --des Exp --itr $ITR

# iTransformer
echo "[Electricity] iTransformer T=720"
python3 -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id electricity_96_720 --model iTransformer --data custom \
    --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 720 \
    --e_layers $E_LAYERS --d_layers $D_LAYERS \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --d_model $D_MODEL --d_ff $D_FF \
    --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
    --use_amp --des Exp --itr $ITR

# DLinear
echo "[Electricity] DLinear T=720"
python3 -u run.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/electricity/ --data_path electricity.csv \
    --model_id electricity_96_720 --model DLinear --data custom \
    --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 720 \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --batch_size 16 --train_epochs $EPOCHS \
    --des Exp --itr $ITR

echo ""
echo "=========================================="
echo " Part 2: Traffic T=192, 336, 720"
echo "=========================================="

# FTMamba
for P in 192 336 720; do
    echo "[Traffic] FTMamba T=$P"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/traffic/ --data_path traffic.csv \
        --model_id traffic_96_${P} --model FTMamba --data custom \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 862 --dec_in 862 --c_out 862 \
        --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND \
        --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --des Exp --itr $ITR
done

# PatchTST
for P in 192 336 720; do
    echo "[Traffic] PatchTST T=$P"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/traffic/ --data_path traffic.csv \
        --model_id traffic_96_${P} --model PatchTST --data custom \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 862 --dec_in 862 --c_out 862 \
        --d_model $D_MODEL --d_ff $D_FF \
        --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --des Exp --itr $ITR
done

# iTransformer
for P in 192 336 720; do
    echo "[Traffic] iTransformer T=$P"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/traffic/ --data_path traffic.csv \
        --model_id traffic_96_${P} --model iTransformer --data custom \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --e_layers $E_LAYERS --d_layers $D_LAYERS \
        --enc_in 862 --dec_in 862 --c_out 862 \
        --d_model $D_MODEL --d_ff $D_FF \
        --batch_size 16 --dropout $DROPOUT --train_epochs $EPOCHS \
        --use_amp --des Exp --itr $ITR
done

# DLinear
for P in 192 336 720; do
    echo "[Traffic] DLinear T=$P"
    python3 -u run.py --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/traffic/ --data_path traffic.csv \
        --model_id traffic_96_${P} --model DLinear --data custom \
        --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $P \
        --enc_in 862 --dec_in 862 --c_out 862 \
        --batch_size 16 --train_epochs $EPOCHS \
        --des Exp --itr $ITR
done

echo ""
echo "=========================================="
echo " Done! Total: 4 + 12 = 16 runs"
echo "=========================================="
