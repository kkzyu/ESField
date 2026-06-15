#!/bin/bash
for z in 10 64;
do
for j in $(seq 10 10 100);
do
for i in $(seq 100 100 900);
do
python sample_mols.py --checkpoint checkpoints/model_updates_738999.ckpt --n-molecules $z --n-atoms $j  --output-dir raw_mols --device cuda --start_t $i --beta 25
done
done
done

