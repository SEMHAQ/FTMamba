# Reviewer Experiment Suite for Symmetry (PowerShell)
# Run: .\run_reviewer_experiments.ps1
# Auto-skips experiments already in results/ directory

$CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_VISIBLE_DEVICES = "0"

$SEQ_LEN = 96; $LABEL_LEN = 48; $E_LAYERS = 3; $D_LAYERS = 1
$D_MODEL = 512; $D_FF = 64; $D_CONV = 4; $EXPAND = 2; $DROPOUT = 0.1
$N_HEADS = 8; $ITR = 1; $BATCH_SIZE = 64

function Invoke-Exp {
    param($Cmd, $Label)
    # Parse label to extract model, dataset, pred_len for skip check
    # Label format: "ModelName on DatasetName (pred_len)" or "mode on DatasetName (T=pred_len)"
    $parts = $Label -split ' '
    $model = $parts[0]
    $dataset = if ($parts.Count -ge 3) { $parts[2] -replace '[()]','' } else { "" }
    $plmatch = [regex]::Match($Label, '(\d+)(?:,|\b|\))')
    if (-not $plmatch.Success) { $plmatch = [regex]::Match($Label, 'T[= ]*(\d+)') }
    $pl = if ($plmatch.Success) { $plmatch.Groups[1].Value } else { "??" }
    $filter = "long_term_forecast_${dataset}_*_${model}_*_sl${SEQ_LEN}_*_pl${pl}_*_Exp_*"
    Write-Debug "Filter: $filter"
    if (Test-Path results) {
        $existing = Get-ChildItem results -Directory -Filter $filter -ErrorAction SilentlyContinue
        if ($existing -and $existing.Count -gt 0) {
            Write-Host "  [SKIP] $Label"
            return
        }
    }
    Write-Host "  [RUN]  $Label"
    Invoke-Expression $Cmd
}

Write-Host "=========================================="
Write-Host " Reviewer Experiment Suite for Symmetry"
Write-Host "=========================================="

# ============================================================
# 1. Additional baselines (FEDformer, FreTS, S_Mamba, TimeMachine)
# ============================================================
Write-Host "`n=== Phase 1: New Baselines on ETT ==="

$datasets = @("ETTh1", "ETTh2", "ETTm1")
$horizons = @(96, 192, 336, 720)
$model_groups = @(
    @{name="FEDformer"; layers=2; d_ff=2048; bs=64; extra="--n_heads $N_HEADS"},
    @{name="FreTS"; layers=2; d_ff=2048; bs=64; extra="--channel_independence 1"},
    @{name="S_Mamba"; layers=3; d_ff=64; bs=16; extra=""},
    @{name="TimeMachine"; layers=3; d_ff=64; bs=64; extra=""}
)

foreach ($ds in $datasets) {
    foreach ($mg in $model_groups) {
        foreach ($pl in $horizons) {
            $label = "$($mg.name) on $ds ($pl, bs=$($mg.bs))"
            $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${pl}_${pl} --model $($mg.name) --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $($mg.layers) --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $($mg.d_ff) --dropout $DROPOUT --batch_size $($mg.bs) --des Review_Exp --itr $ITR $($mg.extra)"
            Invoke-Exp -Cmd $cmd -Label $label
        }
    }
}
Write-Host "Phase 1 ETT done."

# ============================================================
# 1b. New baselines on Weather (skip FTMamba; S_Mamba bs=8, others 128)
# ============================================================
Write-Host "`n=== Phase 1b: New Baselines on Weather ==="
foreach ($mg in $model_groups) {
    if ($mg.name -eq "FTMamba") { continue }
    $bs = if ($mg.name -eq "S_Mamba") { 8 } else { 128 }
    foreach ($pl in $horizons) {
        $label = "$($mg.name) on Weather ($pl, bs=$bs)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model $($mg.name) --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $($mg.layers) --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $($mg.d_ff) --dropout $DROPOUT --batch_size $bs --des Review_Exp --itr $ITR $($mg.extra)"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}
Write-Host "Phase 1b Weather done."

# ============================================================
# 2. Additional datasets (Electricity, Traffic)
# ============================================================
Write-Host "`n=== Phase 2: Electricity + Traffic ==="
$base_models = @("FTMamba", "PatchTST", "iTransformer", "Mamba", "DLinear", "TimesNet", "Transformer")

# Electricity (321 variates, bs=16)
Write-Host "--- Electricity ---"
foreach ($m in $base_models) {
    $extra = ""; if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" } elseif ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        $label = "$m on Electricity ($pl)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 321 --dec_in 321 --c_out 321 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des Dataset_Exp --itr $ITR $extra"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}

# Traffic (862 variates, bs=8)
Write-Host "--- Traffic ---"
foreach ($m in $base_models) {
    $extra = ""; if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" } elseif ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        $label = "$m on Traffic ($pl)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_${pl}_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 862 --dec_in 862 --c_out 862 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 8 --des Dataset_Exp --itr $ITR $extra"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}
Write-Host "Phase 2 done."

# ============================================================
# 3. Extended ablation study
# ============================================================
Write-Host "`n=== Phase 3: Extended Ablation ==="
$ablation_modes = @("full", "no_freq", "add_fusion", "pure_mamba", "freq_only", "scalar_gate", "channel_gate", "patch_gate")
$horizon_modes = @("full", "no_freq", "add_fusion", "pure_mamba")

# 3a. Multi-dataset ablation (ETT, T=96)
Write-Host "--- Multi-dataset (T=96) ---"
foreach ($ds in $datasets) {
    foreach ($mode in $ablation_modes) {
        $label = "$mode on $ds (T=96)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${mode}_96 --model FTMamba --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}

# 3b. Weather ablation (T=96, bs=8 — FTMamba OOMs at higher bs)
Write-Host "--- Weather (T=96, bs=8) ---"
foreach ($mode in $ablation_modes) {
    $label = "$mode on Weather (T=96)"
    $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${mode}_96 --model FTMamba --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len 96 --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 8 --ablation_mode $mode --des Ablation_Exp --itr 1"
        Invoke-Exp -Cmd $cmd -Label $label
}

# 3c. Horizon-stratified ablation (ETTh1)
Write-Host "--- Horizon-stratified (ETTh1) ---"
foreach ($pl in $horizons) {
    foreach ($mode in $horizon_modes) {
        $label = "$mode on ETTh1 (T=$pl)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_${mode}_${pl} --model FTMamba --data ETTh1 --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size $BATCH_SIZE --ablation_mode $mode --des Ablation_Exp --itr 1"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}
Write-Host "Phase 3 done."

# ============================================================
# 4. Multi-seed PatchTST (3 seeds)
# ============================================================
Write-Host "`n=== Phase 4: Multi-seed PatchTST (itr=3) ==="

foreach ($ds in $datasets) {
    foreach ($pl in $horizons) {
        $label = "PatchTST on $ds ($pl, itr=3)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/$ds/ --data_path $ds.csv --model_id ${ds}_${pl}_${pl} --model PatchTST --data $ds --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 7 --dec_in 7 --c_out 7 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size $BATCH_SIZE --des Multiseed_Exp --itr 3 --n_heads $N_HEADS"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}
foreach ($pl in $horizons) {
    $label = "PatchTST on Weather ($pl, itr=3)"
    $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_${pl}_${pl} --model PatchTST --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff 2048 --dropout $DROPOUT --batch_size 16 --des Multiseed_Exp --itr 3 --n_heads $N_HEADS"
    Invoke-Exp -Cmd $cmd -Label $label
}
Write-Host "Phase 4 done."

# ============================================================
# 5. Controlled batch size on Weather (skip FTMamba)
# ============================================================
Write-Host "`n=== Phase 5: Controlled Batch Size (Weather, bs=16, no FTMamba) ==="

$phase5_models = @("FTMamba", "PatchTST", "iTransformer", "Mamba", "DLinear", "TimesNet", "Transformer")
foreach ($m in $phase5_models) {
    if ($m -eq "FTMamba") { Write-Host "  [SKIP] FTMamba on Weather — OOM"; continue }
    $extra = ""; if ($m -in @("PatchTST","iTransformer","Transformer")) { $extra = "--n_heads $N_HEADS" } elseif ($m -eq "TimesNet") { $extra = "--top_k 5 --num_kernels 6 --n_heads $N_HEADS" }
    foreach ($pl in $horizons) {
        $label = "$m on Weather ($pl, bs=16)"
        $cmd = "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_bs16_${pl} --model $m --data custom --features M --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $pl --e_layers $E_LAYERS --d_layers $D_LAYERS --enc_in 21 --dec_in 21 --c_out 21 --d_model $D_MODEL --d_ff $D_FF --d_conv $D_CONV --expand $EXPAND --dropout $DROPOUT --batch_size 16 --des WeatherBS_Exp --itr $ITR $extra"
        Invoke-Exp -Cmd $cmd -Label $label
    }
}
Write-Host "Phase 5 done."

Write-Host "`n=========================================="
Write-Host " All reviewer experiments completed!"
Write-Host " Run: python collect_results.py > results_reviewer.txt"
Write-Host "=========================================="
