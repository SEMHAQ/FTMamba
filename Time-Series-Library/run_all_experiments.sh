#!/bin/bash
# Master script: Run all baselines + FTMamba on ETTh1, ETTh2, ETTm1, Weather
# Usage: bash run_all_experiments.sh
# Requires: GPU with CUDA, PyTorch, and dependencies installed

set -e

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Configuration
# ============================================================
SEQ_LEN=96
LABEL_LEN=48
PRED_LENS="96 192 336 720"
E_LAYERS=3
D_LAYERS=1
D_MODEL=512
D_FF=64
BATCH_SIZE=64
D_CONV=4
EXPAND=2
DROPOUT=0.1
N_HEADS=8
FACTOR=3
ITR=1

# Models to compare
MODELS=("FTMamba" "PatchTST" "iTransformer" "Mamba" "DLinear" "TimesNet" "Transformer")

# Datasets: name, path, enc_in, root
declare -A DATASET_ROOT DATASET_FILE DATASET_ENC_IN DATASET_TYPE
DATASET_ROOT[ETTh1]="dataset/ETT-small"
DATASET_FILE[ETTh1]="ETTh1.csv"
DATASET_ENC_IN[ETTh1]=7
DATASET_TYPE[ETTh1]="ETTh1"

DATASET_ROOT[ETTh2]="dataset/ETT-small"
DATASET_FILE[ETTh2]="ETTh2.csv"
DATASET_ENC_IN[ETTh2]=7
DATASET_TYPE[ETTh2]="ETTh2"

DATASET_ROOT[ETTm1]="dataset/ETT-small"
DATASET_FILE[ETTm1]="ETTm1.csv"
DATASET_ENC_IN[ETTm1]=7
DATASET_TYPE[ETTm1]="ETTm1"

DATASET_ROOT[Weather]="dataset/weather"
DATASET_FILE[Weather]="weather.csv"
DATASET_ENC_IN[Weather]=21
DATASET_TYPE[Weather]="custom"

DATASETS=("ETTh1" "ETTh2" "ETTm1" "Weather")

# ============================================================
# Run experiments
# ============================================================
echo "=========================================="
echo " FTMamba Experiment Suite"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Pred lengths: ${PRED_LENS}"
echo ""

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for pred_len in ${PRED_LENS}; do
      echo "----------------------------------------"
      echo " Running: ${model} on ${dataset} (pred_len=${pred_len})"
      echo "----------------------------------------"

      # Set model-specific parameters
      EXTRA_ARGS=""
      if [ "$model" = "PatchTST" ]; then
        EXTRA_ARGS="--n_heads ${N_HEADS} --factor ${FACTOR}"
      elif [ "$model" = "iTransformer" ]; then
        EXTRA_ARGS="--n_heads ${N_HEADS} --factor ${FACTOR}"
      elif [ "$model" = "TimesNet" ]; then
        EXTRA_ARGS="--top_k 5 --num_kernels 6 --n_heads ${N_HEADS}"
      elif [ "$model" = "Transformer" ]; then
        EXTRA_ARGS="--n_heads ${N_HEADS} --factor ${FACTOR}"
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
        --d_model ${D_MODEL} \
        --d_ff ${D_FF} \
        --d_conv ${D_CONV} \
        --expand ${EXPAND} \
        --dropout ${DROPOUT} \
        --batch_size ${BATCH_SIZE} \
        --des 'FTMamba_Exp' \
        --itr ${ITR} \
        ${EXTRA_ARGS}

      echo " Done: ${model} on ${dataset} (pred_len=${pred_len})"
      echo ""
    done
  done
done

echo "=========================================="
echo " All experiments completed!"
echo "=========================================="
echo "Results saved in ./results/"
echo "Check the result files for MSE/MAE metrics."
