# Reviewer Experiment Suite for Symmetry (PowerShell)
# Run: .\run_reviewer_experiments.ps1
# Requires: GPU, Python, PyTorch, datasets downloaded

$CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_VISIBLE_DEVICES = "0"

$SEQ_LEN = 96
$LABEL_LEN = 48
$E_LAYERS = 3
$D_LAYERS = 1
$D_MODEL = 512
$D_FF = 64
$D_CONV = 4
$EXPAND = 2
$DROPOUT = 0.1
$N_HEADS = 8
$ITR = 1
$BATCH_SIZE = 64

Write-Host "=========================================="
Write-Host " Reviewer Experiment Suite for Symmetry"
Write-Host "=========================================="

# ============================================================
# 1. Additional baselines
# ============================================================
Write-Host "`n=== Phase 1: Additional Baselines ==="

$datasets = @("ETTh1", "ETTh2", "ETTm1")
$horizons = @(96, 192, 336, 720)
$model_groups = @(
    @{name="FEDformer"; layers=2; d_ff=2048; extra="--n_heads $N_HEADS"},
    @{name="FreTS"; layers=2; d_ff=2048; extra="--channel_independence 1"},
    @{name="S_Mamba"; layers=3; d_ff=64; extra=""},
    @{name="TimeMachine"; layers=3; d_ff=2048; extra="--n_heads $N_HEADS"}
)

foreach ($ds in $datasets) {
    foreach ($mg in $model_groups) {
        foreach ($pl in $horizons) {
            $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${pl}_${pl} --model $($mg.name) --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $($mg.layers) --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $($mg.d_ff) --dropout $DROPOUT --batch_size $BATCH_SIZE --des Review_Exp --itr $ITR $($mg.extra)"
            Write-Host "$($mg.name) on $ds ($pl)"
            Invoke-Expression $cmd
        }
    }
}

# Weather (21 variates, bs=16)
foreach ($mg in $model_groups) {
    foreach ($pl in $horizons) {
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model $($mg.name) --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $($mg.layers) --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $($mg.d_ff) --dropout $DROPOUT --batch_size 16 --des Review_Exp --itr $ITR $($mg.extra)"
        Write-Host "$($mg.name) on Weather ($pl)"
        Invoke-Expression $cmd
    }
}
Write-Host "Phase 1 complete."

# ============================================================
# 2. Additional datasets (Electricity, Traffic)
# ============================================================
Write-Host "`n=== Phase 2: Additional Datasets ==="

$models = @("FTMamba", "PatchTST", "iTransformer", "Mamba", "DLinear", "TimesNet", "Transformer")

# Electricity
foreach ($m in $models) {
    $extra = ""
    if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" }
    if ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        Write-Host "$m on Electricity ($pl)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 321 --dec_in 321 --c_out 321 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des Dataset_Exp --itr $ITR $extra
    }
}

# Traffic
foreach ($m in $models) {
    $extra = ""
    if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" }
    if ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        Write-Host "$m on Traffic ($pl)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 862 --dec_in 862 --c_out 862 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 8 --des Dataset_Exp --itr $ITR $extra
    }
}
Write-Host "Phase 2 complete."

# ============================================================
# 3. Extended ablation
# ============================================================
Write-Host "`n=== Phase 3: Extended Ablation ==="

$ablation_modes = @("full", "no_freq", "add_fusion", "pure_mamba", "freq_only", "scalar_gate", "channel_gate", "patch_gate")

# Multi-dataset ablation (T=96)
foreach ($ds in $datasets) {
    foreach ($mode in $ablation_modes) {
        Write-Host "$mode on $ds (T=96)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${mode}_96 --model FTMamba --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1
    }
}

# Weather ablation
foreach ($mode in $ablation_modes) {
    Write-Host "$mode on Weather (T=96)"
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${mode}_96 --model FTMamba --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --ablation_mode $mode --des Ablation_Exp --itr 1
}

# Horizon-stratified ablation (ETTh1)
$horizon_modes = @("full", "no_freq", "add_fusion", "pure_mamba")
foreach ($pl in $horizons) {
    foreach ($mode in $horizon_modes) {
        Write-Host "$mode on ETTh1 (T=$pl)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_${mode}_${pl} --model FTMamba --data ETTh1 --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1
    }
}
Write-Host "Phase 3 complete."

# ============================================================
# 4. Multi-seed PatchTST (3 seeds)
# ============================================================
Write-Host "`n=== Phase 4: Multi-seed Baselines ==="

foreach ($ds in $datasets) {
    foreach ($pl in $horizons) {
        Write-Host "PatchTST on $ds ($pl, itr=3)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${pl}_${pl} --model PatchTST --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size $BATCH_SIZE --des Multiseed_Exp --itr 3 --n_heads $N_HEADS
    }
}
foreach ($pl in $horizons) {
    Write-Host "PatchTST on Weather ($pl, itr=3)"
    python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model PatchTST --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 16 --des Multiseed_Exp --itr 3 --n_heads $N_HEADS
}
Write-Host "Phase 4 complete."

# ============================================================
# 5. Controlled batch size on Weather
# ============================================================
Write-Host "`n=== Phase 5: Controlled Batch Size (Weather, bs=16) ==="

foreach ($m in $models) {
    $extra = ""
    if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" }
    if ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        Write-Host "$m on Weather ($pl, bs=16)"
        python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_bs16_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des WeatherBS_Exp --itr $ITR $extra
    }
}
Write-Host "Phase 5 complete."

Write-Host "`n=========================================="
Write-Host " All reviewer experiments completed!"
Write-Host "=========================================="
Write-Host "Run: python collect_results.py > results_reviewer.txt"
