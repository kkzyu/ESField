#!/bin/bash

is_float() {
  [[ $1 =~ ^-?[0-9]*\.?[0-9]+$ ]]
}

dataset_path="../edm/dataset/pdbbind"
output_path="./results/guidance"

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


#for x in $(ls ../edm/dataset/pdbbind); do
test=( 5zcu )
for x in ${test[@]}; do
  out_path="${output_path}/${x}";
  mkdir -p $out_path;
  n_atoms=$(python ../edm/src/utils/count_atoms.py $dataset_path/$x/ligand.sdf)
  i=0;
  echo "----------";
  echo "Generating ligands for protein ${x}";
  
  # optimize mols
  echo " --> Optimizing molecules with FF";
  echo "   - Copying generated_mols.sdf to mols_raw.sdf...";
  cp $out_path/generated_mols.sdf $out_path/mols_raw.sdf
  echo "   - Running postoptimization.py...";
  python  postoptimization.py "${dataset_path}/${x}/protein.pdb" "${out_path}/mols_raw.sdf" "${out_path}/opt_mols_raw.sdf";
  echo "   - Optimization complete.";
  
  # keep only largest fragments
  echo "   - Removing fragments from mols_raw.sdf...";
  python src/guidance_plugins/utils/remove_fragments.py "${out_path}/mols_raw.sdf" "${out_path}/mols.sdf";
  echo "   - Removing fragments from opt_mols_raw.sdf...";
  python src/guidance_plugins/utils/remove_fragments.py "${out_path}/opt_mols_raw.sdf" "${out_path}/opt_mols.sdf";

  # saving figures
  echo " --> Saving pngs";
  python ../edm/src/utils/plot_mols.py "${dataset_path}/${x}/ligand.sdf" "${out_path}/mols.sdf" "${out_path}/mols.png";
  python ../edm/src/utils/plot_mols.py "${dataset_path}/${x}/ligand.sdf" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols.png";
      
  echo " --> Computing interactions";
  python ../edm/compute_interactions.py "${dataset_path}/${x}/protein.pdb" "${out_path}/mols.sdf" "${out_path}/mols_interactions.csv";
  python ../edm/compute_interactions.py "${dataset_path}/${x}/protein.pdb" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols_interactions.csv";
  
  if [ $use_glide -eq 1 ]; then
    echo " --> Computing docking scores with GLIDE";
    python ../edm/compute_docking_scores.py "${dataset_path}/${x}/grid.zip" "${out_path}/mols.sdf" "${out_path}/mols_docking";
    python ../edm/compute_docking_scores.py "${dataset_path}/${x}/grid.zip" "${out_path}/opt_mols.sdf" "${out_path}/opt_mols_docking";
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
  i=$((i+1));
done

echo ""
echo "=== Computing Summary Metrics ==="

# Check if generate_paper_tables.py exists and run summary
if [ -f "../generate_paper_tables.py" ]; then
    echo "Computing summary metrics..."
    python3 << 'EOF'
import sys
sys.path.append('..')
from generate_paper_tables import get_qed, get_strain_energy, get_interaction_count, get_vina_score
from pathlib import Path
import numpy as np

# Check which SDF files exist
sdf_files = [
    Path('./results/guidance/5zcu/mols.sdf'),
    Path('./results/guidance/5zcu/opt_mols.sdf')
]

found_file = False
for sdf_file in sdf_files:
    if sdf_file.exists():
        found_file = True
        print(f'Processing: {sdf_file.name}')
        
        # Get Vina scores
        vina_results = get_vina_score(sdf_file)
        if vina_results['scores']:
            scores = np.array(vina_results['scores'])
            print(f'  Vina - Mean: {np.mean(scores):.3f}, Median: {np.median(scores):.3f}, Neg ratio: {vina_results["neg_ratio"]:.3f}')
        
        # Get QED
        qed_results = get_qed(sdf_file)
        if qed_results['qeds']:
            qed_mean = np.mean(qed_results['qeds'])
            print(f'  QED: {qed_mean:.3f}')
        
        # Valid ratio
        if qed_results['protein_counter'] > 0:
            # Count total molecules from SDF file directly instead of using protein_counter * 10
            try:
                from rdkit import Chem
                suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
                total_molecules = len([mol for mol in suppl if mol is not None])
                valid_molecules = len(qed_results['qeds'])
                valid_ratio = (valid_molecules / total_molecules) * 100 if total_molecules > 0 else 0
                print(f'  Valid: {valid_ratio:.1f}%')
            except Exception as e:
                # Fallback to original calculation if RDKit fails
                valid_ratio = len(qed_results['qeds']) / max(1, qed_results['protein_counter']) * 100
                print(f'  Valid: {valid_ratio:.1f}%')
        else:
            print('  Valid: 0.0%')
        
        # Interactions
        interaction_results = get_interaction_count(sdf_file)
        print(f'  Interactions: {interaction_results["interaction_count"]}')
        
        # Strain energy
        strain_results = get_strain_energy(sdf_file)
        if strain_results['strain_energy']:
            strain_mean = np.nanmean(strain_results['strain_energy'])
            print(f'  Strain Energy: {strain_mean:.3f}')
        
        break

if not found_file:
    print('Warning: No SDF files found. Results saved to files.')
EOF
else
    echo "Warning: generate_paper_tables.py not found. Results saved to files."
fi

echo ""
echo "=== Postprocessing Complete! ==="
echo "Results saved in: ${output_path}/"
echo ""
echo "To check individual results:"
echo "ls -la ${output_path}/*/"
