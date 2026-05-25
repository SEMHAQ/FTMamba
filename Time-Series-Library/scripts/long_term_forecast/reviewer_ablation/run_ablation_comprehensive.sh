#!/bin/bash
# Extended ablation study for reviewer R1.5 and R3.1
# 1. Multi-dataset ablation (all 4 datasets, horizon 96)
# 2. Horizon-stratified ablation (ETTh1, all horizons)
# 3. Architectural variants (pure_mamba, freq_only, scalar_gate, etc.)

set -e
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=3
D_LAYERS=1
D_MODEL=512
D_FF=64
BATCH_SIZE=64
D_CONV=4
EXPAND=2
DROPOUT=0.1

# ============================================================
# 1. Multi-dataset ablation (all 4 datasets, T=96)
# ============================================================
echo "===== 1. Multi-dataset Ablation (T=96) ====="
ABLATION_MODES=("full" "no_freq" "add_fusion" "pure_mamba" "freq_only" "scalar_gate" "channel_gate" "patch_gate")

declare -A DATASET_ROOT DATASET_FILE DATASET_ENC_IN DATASET_TYPE
DATASET_ROOT[ETTh1]="dataset/ETT-small"; DATASET_FILE[ETTh1]="ETTh1.csv"; DATASET_ENC_IN[ETTh1]=7; DATASET_TYPE[ETTh1]="ETTh1"
DATASET_ROOT[ETTh2]="dataset/ETT-small"; DATASET_FILE[ETTh2]="ETTh2.csv"; DATASET_ENC_IN[ETTh2]=7; DATASET_TYPE[ETTh2]="ETTh2"
DATASET_ROOT[ETTm1]="dataset/ETT-small"; DATASET_FILE[ETTm1]="ETTm1.csv"; DATASET_ENC_IN[ETTm1]=7; DATASET_TYPE[ETTm1]="ETTm1"
DATASET_ROOT[Weather]="dataset/weather"; DATASET_FILE[Weather]="weather.csv"; DATASET_ENC_IN[Weather]=21; DATASET_TYPE[Weather]="custom"
DATASETS=("ETTh1" "ETTh2" "ETTm1" "Weather")

for dataset in "${DATASETS[@]}"; do
  for mode in "${ABLATION_MODES[@]}"; do
    echo "Ablation: ${mode} on ${dataset} (T=96)"
    BS=$BATCH_SIZE
    [ "$dataset" = "Weather" ] && BS=16

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./${DATASET_ROOT[$dataset]}/ \
      --data_path ${DATASET_FILE[$dataset]} \
      --model_id ${dataset}_${mode}_96 \
      --model FTMamba \
      --data ${DATASET_TYPE[$dataset]} \
      --features M \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len 96 \
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
      --batch_size ${BS} \
      --ablation_mode ${mode} \
      --des 'Ablation_Exp' \
      --itr 1
    echo " Done: ${mode} on ${dataset}"
  done
done

# ============================================================
# 2. Horizon-stratified ablation (ETTh1, T=96,192,336,720)
# ============================================================
echo ""
echo "===== 2. Horizon-stratified Ablation (ETTh1) ====="
PRED_LENS="96 192 336 720"
HORIZON_MODES=("full" "no_freq" "add_fusion" "pure_mamba")

for pred_len in ${PRED_LENS}; do
  for mode in "${HORIZON_MODES[@]}"; do
    echo "Ablation: ${mode} on ETTh1 (T=${pred_len})"
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${mode}_${pred_len} \
      --model FTMamba \
      --data ETTh1 \
      --features M \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${pred_len} \
      --e_layers ${E_LAYERS} \
      --d_layers ${D_LAYERS} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model ${D_MODEL} \
      --d_ff ${D_FF} \
      --d_conv ${D_CONV} \
      --expand ${EXPAND} \
      --dropout ${DROPOUT} \
      --batch_size ${BATCH_SIZE} \
      --ablation_mode ${mode} \
      --des 'Ablation_Exp' \
      --itr 1
    echo " Done: ${mode} on ETTh1 (T=${pred_len})"
  done
done

echo "All ablation experiments completed!"
