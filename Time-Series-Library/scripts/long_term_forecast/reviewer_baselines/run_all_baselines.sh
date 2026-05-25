#!/bin/bash
# Run ALL additional baselines (FEDformer, FreTS, S-Mamba, TimeMachine)
# on ETTh1, ETTh2, ETTm1, Weather × 96,192,336,720

set -e
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=96
LABEL_LEN=48
PRED_LENS="96 192 336 720"
E_LAYERS=2
D_LAYERS=1
D_MODEL=512
D_FF=2048
BATCH_SIZE=64
D_CONV=4
EXPAND=2
DROPOUT=0.1
N_HEADS=8
ITR=1

declare -A DATASET_ROOT DATASET_FILE DATASET_ENC_IN DATASET_TYPE
DATASET_ROOT[ETTh1]="dataset/ETT-small"; DATASET_FILE[ETTh1]="ETTh1.csv"; DATASET_ENC_IN[ETTh1]=7; DATASET_TYPE[ETTh1]="ETTh1"
DATASET_ROOT[ETTh2]="dataset/ETT-small"; DATASET_FILE[ETTh2]="ETTh2.csv"; DATASET_ENC_IN[ETTh2]=7; DATASET_TYPE[ETTh2]="ETTh2"
DATASET_ROOT[ETTm1]="dataset/ETT-small"; DATASET_FILE[ETTm1]="ETTm1.csv"; DATASET_ENC_IN[ETTm1]=7; DATASET_TYPE[ETTm1]="ETTm1"
DATASET_ROOT[Weather]="dataset/weather"; DATASET_FILE[Weather]="weather.csv"; DATASET_ENC_IN[Weather]=21; DATASET_TYPE[Weather]="custom"

DATASETS=("ETTh1" "ETTh2" "ETTm1" "Weather")

# Models: FEDformer, FreTS, S-Mamba, TimeMachine
NEW_MODELS=("FEDformer" "FreTS" "S_Mamba" "TimeMachine")

for dataset in "${DATASETS[@]}"; do
  for model in "${NEW_MODELS[@]}"; do
    for pred_len in ${PRED_LENS}; do
      echo "Running: ${model} on ${dataset} (pred_len=${pred_len})"

      # Model-specific args
      EXTRA_ARGS=""
      if [ "$model" = "FEDformer" ]; then
        EXTRA_ARGS="--n_heads ${N_HEADS} --version Fourier --mode_select random --modes 64"
      elif [ "$model" = "FreTS" ]; then
        EXTRA_ARGS="--channel_independence 1"
      elif [ "$model" = "TimeMachine" ]; then
        EXTRA_ARGS="--d_ff ${D_FF} --n_heads ${N_HEADS}"
      fi

      # S_Mamba and FreTS use d_model differently
      MODEL_D_MODEL=$D_MODEL
      MODEL_D_FF=$D_FF
      if [ "$model" = "S_Mamba" ]; then
        MODEL_D_FF=64  # S-Mamba uses smaller d_ff
      fi

      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./${DATASET_ROOT[$dataset]}/ \
        --data_path ${DATASET_FILE[$dataset]} \
        --model_id ${dataset}_${pred_len}_${pred_len} \
        --model $model \
        --data ${DATASET_TYPE[$dataset]} \
        --features M \
        --seq_len ${SEQ_LEN} \
        --label_len ${LABEL_LEN} \
        --pred_len ${pred_len} \
        --e_layers ${E_LAYERS} \
        --d_layers ${D_LAYERS} \
        --enc_in ${DATASET_ENC_IN[$dataset]} \
        --dec_in ${DATASET_ENC_IN[$dataset]} \
        --c_out ${DATASET_ENC_IN[$dataset]} \
        --d_model ${MODEL_D_MODEL} \
        --d_ff ${MODEL_D_FF} \
        --dropout ${DROPOUT} \
        --batch_size ${BATCH_SIZE} \
        --des 'Reviewer_Exp' \
        --itr ${ITR} \
        ${EXTRA_ARGS}

      echo " Done: ${model} on ${dataset} (pred_len=${pred_len})"
    done
  done
done

echo "All additional baselines completed!"
