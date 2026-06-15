#!/bin/bash
export PYTHONPATH="/root/baselines/DecompDiff/code/DecompDiff-main:/root/ESField/src:$PYTHONPATH"
cd /root/baselines/DecompDiff/code/DecompDiff-main
exec python3 "$@"
