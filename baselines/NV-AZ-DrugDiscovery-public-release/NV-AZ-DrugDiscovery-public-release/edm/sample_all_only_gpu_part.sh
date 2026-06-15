#!/bin/bash

is_float() {
  [[ $1 =~ ^-?[0-9]*\.?[0-9]+$ ]]
}

NMOLECULES=16
BETA=10.0
# BETA="100.0"

#dataset_path="../../../Datasets/PDBBindOriginalCleaned/cleaned_dataset";
dataset_path="../../Datasets/PDBBind"
output_path="generated_250303_bb"

OPTIONS=$(getopt -o g --long use-glide -- "$@")
# OPTIONS=$(getopt -o g,f: --long use-glide,file: -- "$@")
# $? checks if getopt failed
if [ $? -ne 0 ]; then
  echo "Incorrect options provided"
  exit 1
fi

# This line is necessary to reorder the arguments as per the getopt result
eval set -- "$OPTIONS"

use_glide=0

# Parse the options
while true; do
  case "$1" in
    -g | --use-glide)
      use_glide=1
      shift ;;
    #-f | --file)
    #  filename="$2"
    #  shift 2 ;;
    --)
      shift
      break ;;
    *)
      echo "Invalid option: $1"
      exit 1 ;;
  esac
done


declare -a steps=(0 100 250 500);

if is_float "$BETA"; then
  # Convert to string with two decimal digits
  BETA=$(printf "%.2f" "$BETA")
fi

# for x in $(seq 1 1); do
#  x="5zcu";
#  x="6qrd";
for x in $(ls dataset/pdbbind); do
  n_atoms=$(python src/utils/count_atoms.py dataset/pdbbind/$x/ligand.sdf);
  i=0;
  echo "----------";
  echo "Generating ligands for protein ${x}";
  for step in "${steps[@]}"; do 
    echo " --> Guidance starting at step: ${step}";
    out_path="${output_path}/${x}/${step}";
    mkdir -p $out_path;
    npy_trj_prefix="${NMOLECULES}_${n_atoms}_${BETA}_000";
    if [ ! -f "${out_path}/${npy_trj_prefix}.npy" ]; then
      if [ "$i" -eq 0 ]; then
        python conditional_sample_mols.py --checkpoint checkpoints/conditional_model_updates_487_epochs.ckpt --protein "dataset/pdbbind/${x}"  --n-molecules $NMOLECULES --n-atoms $n_atoms --output-dir $out_path --beta=$BETA --guidance-time=$step --device cuda --x0-guidance
      else
        python conditional_sample_mols.py --checkpoint checkpoints/conditional_model_updates_487_epochs.ckpt --protein "dataset/pdbbind/${x}"  --n-molecules $NMOLECULES --n-atoms $n_atoms --output-dir $out_path --beta=$BETA --guidance-time=$step --device cuda --x0-guidance --xt "${output_path}/${x}/0/${npy_trj_prefix}_trj.npy"
      fi;
      
    fi;
    i=$((i+1));
  done;
done;
