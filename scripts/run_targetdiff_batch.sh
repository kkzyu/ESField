#!/bin/bash
# Comprehensive TargetDiff + ESField experiment
# Pockets: 3mfw, 6o4x
# Conditions: unguided, hard_fix, kinematic
# Molecules per condition: 50
# Steps: 500

set -e
cd /root/ESField
export PYTHONPATH=src

OUTDIR="experiments/targetdiff_replication"

echo "============================================"
echo "TargetDiff Full Experiment"
echo "Date: $(date)"
echo "============================================"

for POCKET in 3mfw 6o4x; do
  for MODE in unguided hard_fix kinematic; do
    echo ""
    echo ">>> $POCKET / $MODE ($(date))"
    python scripts/run_targetdiff_full_pipeline.py \
      --pocket $POCKET \
      --mode $MODE \
      --n-samples 50 \
      --num-steps 500 \
      --batch-size 8 \
      --output-dir $OUTDIR
    echo "<<< $POCKET / $MODE done ($(date))"
  done
done

# Generate summary
echo ""
echo "============================================"
echo "Generating summary"
echo "============================================"
python -c "
import json
from pathlib import Path
outdir = Path('$OUTDIR')
for pocket in ['3mfw', '6o4x']:
    summary = outdir / pocket / 'summary.json'
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
        print(f'\n{pocket}:')
        for mode, s in data['conditions'].items():
            print(f'  {mode:15s} DirectOcc={s[\"direct_occ\"]:.1%}  '
                  f'KPE_ratio={s[\"kpe_ratio\"]:.6f}  '
                  f'Valid={s[\"n_valid\"]}/{s[\"n_total\"]}')
"

echo ""
echo "Done at $(date)"
