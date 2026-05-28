#!/bin/bash
# Reviewer Experiment Suite for Symmetry (Bash / WSL2)
# Run: bash run_reviewer_experiments.sh
# Auto-skips experiments already in result_long_term_forecast.txt

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
N_HEADS=8
ITR=1
BATCH_SIZE=64

RESULT_FILE="result_long_term_forecast.txt"

invoke_exp() {
    local cmd="$1"
    local label="$2"

    # Parse: "ModelName on DatasetName (pred_len, ...)"
    local model dataset pl
    model=$(echo "$label" | awk '{print $1}')
    dataset=$(echo "$label" | awk '{print $3}')
    # Extract prediction length: first number in parentheses
    pl=$(echo "$label" | grep -oP '\(\K\d+' | head -1)
    if [ -z "$pl" ]; then
        pl=$(echo "$label" | grep -oP 'T[= ]*\K\d+' | head -1)
    fi

    # Check if already done
    if [ -f "$RESULT_FILE" ]; then
        if grep -qE "forecast_${dataset}_.*_${model}_.*_pl${pl}_.*_Exp_" "$RESULT_FILE" 2>/dev/null; then
            echo "  [SKIP] $label"
            return
        fi
    fi

    echo "  [RUN]  $label"
    eval "$cmd"
}

echo "=========================================="
echo " Reviewer Experiment Suite for Symmetry"
echo "=========================================="

# ============================================================
# 1. Additional baselines on ETT
# ============================================================
echo ""
echo "=== Phase 1: New Baselines on ETT ==="

DATASETS=("ETTh1" "ETTh2" "ETTm1")
HORIZONS=(96 192 336 720)

for ds in "${DATASETS[@]}"; do
    for pl in "${HORIZONS[@]}"; do

        # FEDformer
        label="FEDformer on $ds ($pl, bs=64)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${pl}_${pl} --model FEDformer --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers 2 --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 64 --des Review_Exp --itr $ITR --n_heads $N_HEADS"
        invoke_exp "$cmd" "$label"

        # FreTS
        label="FreTS on $ds ($pl, bs=64)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${pl}_${pl} --model FreTS --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers 2 --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 64 --des Review_Exp --itr $ITR --channel_independence 1"
        invoke_exp "$cmd" "$label"

        # S_Mamba (bs=16 to avoid OOM)
        label="S_Mamba on $ds ($pl, bs=16)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${pl}_${pl} --model S_Mamba --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --dropout $DROPOUT --batch_size 16 --des Review_Exp --itr $ITR"
        invoke_exp "$cmd" "$label"

        # TimeMachine
        label="TimeMachine on $ds ($pl, bs=64)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${pl}_${pl} --model TimeMachine --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --dropout $DROPOUT --batch_size 64 --des Review_Exp --itr $ITR"
        invoke_exp "$cmd" "$label"

    done
done
echo "Phase 1 ETT done."

# ============================================================
# 1b. New baselines on Weather
# ============================================================
echo ""
echo "=== Phase 1b: New Baselines on Weather ==="

for pl in "${HORIZONS[@]}"; do

    label="FEDformer on Weather ($pl, bs=128)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model FEDformer --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers 2 --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 128 --des Review_Exp --itr $ITR --n_heads $N_HEADS"
    invoke_exp "$cmd" "$label"

    label="FreTS on Weather ($pl, bs=128)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model FreTS --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers 2 --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 128 --des Review_Exp --itr $ITR --channel_independence 1"
    invoke_exp "$cmd" "$label"

    label="S_Mamba on Weather ($pl, bs=8)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model S_Mamba --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --dropout $DROPOUT --batch_size 8 --des Review_Exp --itr $ITR"
    invoke_exp "$cmd" "$label"

    label="TimeMachine on Weather ($pl, bs=128)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model TimeMachine --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --dropout $DROPOUT --batch_size 128 --des Review_Exp --itr $ITR"
    invoke_exp "$cmd" "$label"

done
echo "Phase 1b Weather done."

# ============================================================
# 2. Additional datasets (Electricity, Traffic)
# ============================================================
echo ""
echo "=== Phase 2: Electricity + Traffic ==="
BASE_MODELS=("FTMamba" "PatchTST" "iTransformer" "Mamba" "DLinear" "TimesNet" "Transformer")
# Speed flags for large datasets (Electricity/Traffic): AMP + fewer epochs
LARGE_DS_FLAGS="--use_amp --train_epochs 5"

# Electricity (321 variates, bs=32 — fits in 24GB with AMP)
echo "--- Electricity ---"
for m in "${BASE_MODELS[@]}"; do
    EXTRA=""
    if [[ "$m" == "PatchTST" || "$m" == "iTransformer" || "$m" == "Transformer" ]]; then
        EXTRA="--n_heads $N_HEADS"
    elif [[ "$m" == "TimesNet" ]]; then
        EXTRA="--top_k 5 --num_kernels 6 --n_heads $N_HEADS"
    fi
    for pl in "${HORIZONS[@]}"; do
        label="$m on Electricity ($pl)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 321 --dec_in 321 --c_out 321 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 32 --des Dataset_Exp --itr $ITR $LARGE_DS_FLAGS $EXTRA"
        invoke_exp "$cmd" "$label"
    done
done

# Traffic (862 variates, bs=16)
echo "--- Traffic ---"
for m in "${BASE_MODELS[@]}"; do
    EXTRA=""
    if [[ "$m" == "PatchTST" || "$m" == "iTransformer" || "$m" == "Transformer" ]]; then
        EXTRA="--n_heads $N_HEADS"
    elif [[ "$m" == "TimesNet" ]]; then
        EXTRA="--top_k 5 --num_kernels 6 --n_heads $N_HEADS"
    fi
    for pl in "${HORIZONS[@]}"; do
        label="$m on Traffic ($pl)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 862 --dec_in 862 --c_out 862 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des Dataset_Exp --itr $ITR $LARGE_DS_FLAGS $EXTRA"
        invoke_exp "$cmd" "$label"
    done
done
echo "Phase 2 done."

# ============================================================
# 3. Extended ablation study
# ============================================================
echo ""
echo "=== Phase 3: Extended Ablation ==="
ABLATION_MODES=("full" "no_freq" "add_fusion" "pure_mamba" "freq_only" "scalar_gate" "channel_gate" "patch_gate")
HORIZON_MODES=("full" "no_freq" "add_fusion" "pure_mamba")

# 3a. Multi-dataset ablation (T=96)
echo "--- Multi-dataset (T=96) ---"
for ds in "${DATASETS[@]}"; do
    for mode in "${ABLATION_MODES[@]}"; do
        label="$mode on $ds (T=96)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${mode}_96 --model FTMamba --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1"
        invoke_exp "$cmd" "$label"
    done
done

# 3b. Weather ablation (T=96, bs=8)
echo "--- Weather (T=96, bs=8) ---"
for mode in "${ABLATION_MODES[@]}"; do
    label="$mode on Weather (T=96)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${mode}_96 --model FTMamba --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 8 --ablation_mode $mode --des Ablation_Exp --itr 1"
    invoke_exp "$cmd" "$label"
done

# 3c. Horizon-stratified ablation (ETTh1)
echo "--- Horizon-stratified (ETTh1) ---"
for pl in "${HORIZONS[@]}"; do
    for mode in "${HORIZON_MODES[@]}"; do
        label="$mode on ETTh1 (T=$pl)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_${mode}_${pl} --model FTMamba --data ETTh1 --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1"
        invoke_exp "$cmd" "$label"
    done
done
echo "Phase 3 done."

# ============================================================
# 4. Multi-seed PatchTST (itr=3)
# ============================================================
echo ""
echo "=== Phase 4: Multi-seed PatchTST (itr=3) ==="

for ds in "${DATASETS[@]}"; do
    for pl in "${HORIZONS[@]}"; do
        label="PatchTST on $ds ($pl, itr=3)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path ${ds}.csv --model_id ${ds}_${pl}_${pl} --model PatchTST --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size $BATCH_SIZE --des Multiseed_Exp --itr 3 --n_heads $N_HEADS"
        invoke_exp "$cmd" "$label"
    done
done

for pl in "${HORIZONS[@]}"; do
    label="PatchTST on Weather ($pl, itr=3)"
    cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model PatchTST --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 16 --des Multiseed_Exp --itr 3 --n_heads $N_HEADS"
    invoke_exp "$cmd" "$label"
done
echo "Phase 4 done."

# ============================================================
# 5. Controlled batch size on Weather (skip FTMamba)
# ============================================================
echo ""
echo "=== Phase 5: Controlled Batch Size (Weather, bs=16) ==="
PHASE5_MODELS=("FTMamba" "PatchTST" "iTransformer" "Mamba" "DLinear" "TimesNet" "Transformer")

for m in "${PHASE5_MODELS[@]}"; do
    if [[ "$m" == "FTMamba" ]]; then
        echo "  [SKIP] FTMamba on Weather — OOM"
        continue
    fi
    EXTRA=""
    if [[ "$m" == "PatchTST" || "$m" == "iTransformer" || "$m" == "Transformer" ]]; then
        EXTRA="--n_heads $N_HEADS"
    elif [[ "$m" == "TimesNet" ]]; then
        EXTRA="--top_k 5 --num_kernels 6 --n_heads $N_HEADS"
    fi
    for pl in "${HORIZONS[@]}"; do
        label="$m on Weather ($pl, bs=16)"
        cmd="python3 -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_bs16_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des WeatherBS_Exp --itr $ITR $EXTRA"
        invoke_exp "$cmd" "$label"
    done
done
echo "Phase 5 done."

echo ""
echo "=========================================="
echo " All reviewer experiments completed!"
echo " Run: python collect_results.py > results_reviewer.txt"
echo "=========================================="
