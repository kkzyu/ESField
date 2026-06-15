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

#for x in $(seq 1 1); do
#  x="6qrd";
  # x="5zcu";
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
      
      # convert clouds to mols 
      echo " --> Converting point clouds to molecules";
      python src/guidance_plugins/utils/cloud2mol.py "${out_path}/${npy_trj_prefix}.npy" "${out_path}/mols_raw.sdf"
      
      # optimize mols
      echo " --> Optimizing molecules with FF";
      python src/guidance_plugins/local_minimization.py "dataset/pdbbind/${x}/protein.pdb" "${out_path}/mols_raw.sdf" "${out_path}/opt_mols_raw.sdf";

      # keep only largest fragments
      python src/guidance_plugins/utils/remove_fragments.py "${out_path}/mols_raw.sdf" "${out_path}/mols.sdf";
      python src/guidance_plugins/utils/remove_fragments.py "${out_path}/opt_mols_raw.sdf" "${out_path}/opt_mols.sdf";

      # saving figures
      echo " --> Saving pngs";
      python src/utils/plot_mols.py "${dataset_path}/${x}/ligand.sdf" "${out_path}/mols.sdf" "${out_path}/mols.png";
      python src/utils/plot_mols.py "${dataset_path}/${x}/ligand.sdf" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols.png";
      
      echo " --> Computing interactions";
      python compute_interactions.py "${dataset_path}/${x}/protein.pdb" "${out_path}/mols.sdf" "${out_path}/mols_interactions.csv";
      python compute_interactions.py "${dataset_path}/${x}/protein.pdb" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols_interactions.csv";

      if [ $use_glide -eq 1 ]; then
        echo " --> Computing docking scores with GLIDE";
        python compute_docking_scores.py "${dataset_path}/${x}/grid.zip" "${out_path}/mols.sdf" "${out_path}/mols_docking";
        python compute_docking_scores.py "${dataset_path}/${x}/grid.zip" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols_docking";
        echo " --> Done";
      fi
      echo " --> Computing docking scores with VINA";
      obabel "${dataset_path}/${x}/protein.pdb"  -O "${out_path}/protein.pdbqt"  -xr  -p 7.4 --partialcharge eem
      obabel -isdf "${out_path}/mols.sdf" -opdbqt -h -O "${out_path}/mols.pdbqt"
      obabel -isdf "${out_path}/opt_mols.sdf" -opdbqt -h -O "${out_path}/opt_mols.pdbqt"
      mkdir -p "${out_path}/mols_vina"
      mkdir -p "${out_path}/opt_mols_vina"
     
      ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina_split --input "${out_path}/mols.pdbqt" --ligand "${out_path}/mols_vina/ligand"
      ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina_split --input "${out_path}/opt_mols.pdbqt" --ligand "${out_path}/opt_mols_vina/ligand"
      for ligand in ${out_path}/mols_vina/*.pdbqt; do
        scorefilename="${ligand%.*}"_score.txt
        rm -f $scorefilename
        ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina --receptor $out_path/protein.pdbqt --ligand $ligand --score_only | grep "Affinity" > $scorefilename
      done
      for ligand in ${out_path}/opt_mols_vina/*.pdbqt; do
        scorefilename="${ligand%.*}"_score.txt
        rm -f $scorefilename
        ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina --receptor $out_path/protein.pdbqt --ligand $ligand --score_only | grep "Affinity" > $scorefilename
      done
      echo " --> Done";
      echo "----------";
    fi;
    i=$((i+1));
  done;
done;
