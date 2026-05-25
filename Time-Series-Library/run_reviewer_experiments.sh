#!/bin/bash
# Master script: Run ALL reviewer-requested experiments
# Usage: bash run_reviewer_experiments.sh
# Requires: GPU with CUDA, PyTorch, datasets downloaded

set -e
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo " Reviewer Experiment Suite for Symmetry"
echo "=========================================="

# ============================================================
# 1. Additional baselines (FEDformer, FreTS, S-Mamba, TimeMachine)
# ============================================================
echo ""
echo "=== Phase 1: Additional Baselines ==="
bash scripts/long_term_forecast/reviewer_baselines/run_all_baselines.sh

# ============================================================
# 2. Additional datasets (Electricity, Traffic)
# ============================================================
echo ""
echo "=== Phase 2: Additional Datasets ==="
bash scripts/long_term_forecast/reviewer_datasets/run_electricity.sh
bash scripts/long_term_forecast/reviewer_datasets/run_traffic.sh

# ============================================================
# 3. Extended ablation study
# ============================================================
echo ""
echo "=== Phase 3: Extended Ablation ==="
bash scripts/long_term_forecast/reviewer_ablation/run_ablation_comprehensive.sh

# ============================================================
# 4. Multi-seed PatchTST (3 seeds for symmetric comparison)
# ============================================================
echo ""
echo "=== Phase 4: Multi-seed Baselines ==="
bash scripts/long_term_forecast/reviewer_multiseed/run_multiseed_baselines.sh

# ============================================================
# 5. Controlled batch size on Weather
# ============================================================
echo ""
echo "=== Phase 5: Controlled Batch Size (Weather) ==="
bash scripts/long_term_forecast/reviewer_weather/run_weather_controlled.sh

# ============================================================
# 6. Efficiency measurement
# ============================================================
echo ""
echo "=== Phase 6: Efficiency Metrics ==="
python run_efficiency.py

# ============================================================
# 7. Gate visualization (requires trained models)
# ============================================================
echo ""
echo "=== Phase 7: Gate Analysis ==="
python run_gate_analysis.py

# ============================================================
# 8. Collect all results
# ============================================================
echo ""
echo "=== Phase 8: Collect Results ==="
python collect_results.py > results_reviewer.txt
echo "Results saved to results_reviewer.txt"

echo ""
echo "=========================================="
echo " All reviewer experiments completed!"
echo "=========================================="
