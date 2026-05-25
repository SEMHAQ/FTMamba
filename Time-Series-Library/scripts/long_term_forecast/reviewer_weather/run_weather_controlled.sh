#!/bin/bash
# Controlled batch size experiments on Weather
# All models run with batch_size=16 for fair comparison
set -e
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=96
LABEL_LEN=48
PRED_LENS="96 192 336 720"
E_LAYERS=3
D_LAYERS=1
D_MODEL=512
D_FF=64
D_CONV=4
EXPAND=2
DROPOUT=0.1
N_HEADS=8
ITR=1

MODELS=("FTMamba" "PatchTST" "iTransformer" "Mamba" "DLinear" "TimesNet" "Transformer")

echo "===== Weather Controlled Batch Size (batch_size=16) ====="
for model in "${MODELS[@]}"; do
  for pred_len in ${PRED_LENS}; do
    echo "Running: ${model} on Weather (pred_len=${pred_len}, batch_size=16)"

    EXTRA_ARGS=""
    if [ "$model" = "PatchTST" ] || [ "$model" = "iTransformer" ] || [ "$model" = "Transformer" ]; then
      EXTRA_ARGS="--n_heads ${N_HEADS}"
    elif [ "$model" = "TimesNet" ]; then
      EXTRA_ARGS="--top_k 5 --num_kernels 6 --n_heads ${N_HEADS}"
    fi

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id Weather_bs16_${pred_len} \
      --model $model \
      --data custom \
      --features M \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${pred_len} \
      --e_layers ${E_LAYERS} \
      --d_layers ${D_LAYERS} \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --d_model ${D_MODEL} \
      --d_ff ${D_FF} \
      --d_conv ${D_CONV} \
      --expand ${EXPAND} \
      --dropout ${DROPOUT} \
      --batch_size 16 \
      --des 'Weather_BS_Exp' \
      --itr ${ITR} \
      ${EXTRA_ARGS}

    echo " Done"
  done
done
echo "Weather controlled batch size experiments completed!"
