#!/bin/bash
# ESField Four-Step Experiment Pipeline
# Usage: bash scripts/run_all_steps.sh [--step 1|2|3|4] [--gpu CUDA_DEVICE]

set -e
ROOT=/root/ESField
export PYTHONPATH=$ROOT/src
GPU=${GPU:-cuda:0}

echo "========================================"
echo "ESField Experiment Pipeline"
echo "========================================"

# Step 1: Diagnosis (CPU only)
run_step1() {
    echo "[Step 1] Opportunity-Blindness Diagnosis..."
    python $ROOT/scripts/run_step1_diagnosis.py
}

# Step 2: Site Meaningfulness (GPU needed)
run_step2() {
    echo "[Step 2] Site Meaningfulness Validation..."
    echo "  Requires GPU for guided generation."
    echo "  TODO: Implement batch guided generation for correct/random/shuffled sites"
}

# Step 3: ESField Improvement (GPU needed)
run_step3() {
    echo "[Step 3] ESField Improvement..."
    echo "  TODO: batch guided + baseline generation on 20 test pockets"
}

# Step 4: Quality Constraint (GPU needed)
run_step4() {
    echo "[Step 4] Quality-Constrained Validation..."
    echo "  TODO: lambda sweep + quality analysis"
}

case "${1:-}" in
    --step) run_step${2} ;;
    *) run_step1 ;;
esac
