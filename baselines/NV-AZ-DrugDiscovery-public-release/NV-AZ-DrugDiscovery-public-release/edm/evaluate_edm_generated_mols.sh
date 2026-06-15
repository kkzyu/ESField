#!/bin/bash

# Script to analyze all generated molecule files and compute scores
# Run from the edm directory

set -e  # Exit on any error

echo "=== Analyzing All Generated Molecules ==="

# Configuration
DATASET_PATH="dataset/pdbbind/5zcu"
PROTEIN_PDB="${DATASET_PATH}/protein.pdb"
OUTPUT_BASE_DIR="raw_mols"

echo "Using protein: $PROTEIN_PDB"
echo ""

# Define the files to analyze
declare -A FILES=(
    ["unconditional_noguidance_40atoms"]="raw_mols/unconditional_10_40_0.00_0_000.npy"
    ["unconditional_guidance_40atoms"]="raw_mols/unconditional_10_40_10.00_200_000.npy"
    ["conditional_guidance_30atoms"]="raw_mols/conditional_10_30_10.00_200_000.npy"
)

echo "Files to analyze:"
echo "1. unconditional_noguidance_40atoms: ${FILES[unconditional_noguidance_30atoms]} (10 molecules, 30 atoms, unconditional generation without energy guidance, β=10.00)"
echo "2. unconditional_guidance_40atoms: ${FILES[unconditional_guidance_40atoms]} (10 molecules, 40 atoms, unconditional generation with energy guidance, β=10.00)"
echo "3. conditional_guidance_30atoms: ${FILES[conditional_guidance_30atoms]} (10 molecules, 30 atoms, conditional generation with energy guidance, β=10.00)"
echo ""

# Function to compute scores for a given file
compute_scores() {
    local name=$1
    local input_file=$2
    local output_dir="${OUTPUT_BASE_DIR}/${name}"
    
    echo "=== Processing: $name ==="
    echo "Input file: $input_file"
    echo "Output directory: $output_dir"
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file $input_file not found. Skipping..."
        return
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Step 1: Convert point clouds to molecules
    echo "1. Converting point clouds to molecules..."
    python src/guidance_plugins/utils/cloud2mol.py "$input_file" "$output_dir/mols_raw.sdf"
    
    # Step 2: Optimize molecules with force field
    echo "2. Optimizing molecules with force field..."
    python src/guidance_plugins/local_minimization.py "$PROTEIN_PDB" "$output_dir/mols_raw.sdf" "$output_dir/opt_mols_raw.sdf"
    
    # Step 3: Remove fragments (keep only largest fragment)
    echo "3. Removing fragments..."
    python src/guidance_plugins/utils/remove_fragments.py "$output_dir/mols_raw.sdf" "$output_dir/mols.sdf"
    python src/guidance_plugins/utils/remove_fragments.py "$output_dir/opt_mols_raw.sdf" "$output_dir/opt_mols.sdf"
    
    # Step 4: Compute interactions
    echo "4. Computing interactions..."
    python compute_interactions.py "$PROTEIN_PDB" "$output_dir/mols.sdf" "$output_dir/mols_interactions.csv"
    python compute_interactions.py "$PROTEIN_PDB" "$output_dir/opt_mols.sdf" "$output_dir/opt_mols_interactions.csv"
    
    # Step 5: Convert to PDBQT format for Vina
    echo "5. Converting to PDBQT format for Vina..."
    obabel "$PROTEIN_PDB" -O "$output_dir/protein.pdbqt" -xr -p 7.4 --partialcharge eem
    obabel -isdf "$output_dir/mols.sdf" -opdbqt -h -O "$output_dir/mols.pdbqt"
    obabel -isdf "$output_dir/opt_mols.sdf" -opdbqt -h -O "$output_dir/opt_mols.pdbqt"
    
    # Step 6: Compute Vina scores
    echo "6. Computing Vina scores..."
    mkdir -p "$output_dir/mols_vina"
    mkdir -p "$output_dir/opt_mols_vina"
    
    # Split ligands
    ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina_split --input "$output_dir/mols.pdbqt" --ligand "$output_dir/mols_vina/ligand"
    ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina_split --input "$output_dir/opt_mols.pdbqt" --ligand "$output_dir/opt_mols_vina/ligand"
    
    # Score each ligand
    echo "Scoring raw molecules with Vina..."
    for ligand in "$output_dir/mols_vina"/*.pdbqt; do
        scorefilename="${ligand%.*}"_score.txt
        rm -f "$scorefilename"
        ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina --receptor "$output_dir/protein.pdbqt" --ligand "$ligand" --score_only | grep "Affinity" > "$scorefilename"
    done
    
    echo "Scoring optimized molecules with Vina..."
    for ligand in "$output_dir/opt_mols_vina"/*.pdbqt; do
        scorefilename="${ligand%.*}"_score.txt
        rm -f "$scorefilename"
        ./src/guidance_plugins/autodock_vina_1_1_2_linux_x86/bin/vina --receptor "$output_dir/protein.pdbqt" --ligand "$ligand" --score_only | grep "Affinity" > "$scorefilename"
    done
    
    echo "Completed processing: $name"
    echo ""
}

# Process each file
for name in "${!FILES[@]}"; do
    compute_scores "$name" "${FILES[$name]}"
done

# Analyze and compare results
echo "=== Comparing Results ==="

if [ -f "../generate_paper_tables.py" ]; then
    echo "Analyzing Vina scores, molecular quality metrics, and interactions..."
    python -c "
from pathlib import Path
import sys
sys.path.append('..')
from generate_paper_tables import get_vina_score, get_qed, get_posebuster, get_better_than_native, get_interaction_count, get_strain_energy
import numpy as np

results = {}

for name in ['unconditional_noguidance_40atoms', 'unconditional_guidance_40atoms', 'conditional_guidance_30atoms']:
    mols_path = Path(f'raw_mols/{name}/mols.sdf')
    opt_mols_path = Path(f'raw_mols/{name}/opt_mols.sdf')
    
    if mols_path.exists():
        # Vina scores
        vina_results = get_vina_score(mols_path)
        opt_vina_results = get_vina_score(opt_mols_path)
        
        # Molecular quality metrics
        qed_results = get_qed(mols_path)
        pb_results = get_posebuster(mols_path)
        btn_results = get_better_than_native(mols_path, Path('../dataset/pdbbind'))
        interaction_results = get_interaction_count(mols_path)
        strain_results = get_strain_energy(mols_path)
        
        results[name] = {
            'raw_vina': vina_results,
            'opt_vina': opt_vina_results,
            'qed': qed_results,
            'posebuster': pb_results,
            'better_than_native': btn_results,
            'interactions': interaction_results,
            'strain_energy': strain_results
        }
        
        print(f'\\n{name}:')
        
        # Vina scores
        if vina_results['scores']:
            scores = np.array(vina_results['scores'])
            print(f'  Raw Vina - Mean: {np.mean(scores):.3f}, Median: {np.median(scores):.3f}, Neg ratio: {vina_results[\"neg_ratio\"]:.3f}')
        if opt_vina_results['scores']:
            opt_scores = np.array(opt_vina_results['scores'])
            print(f'  Opt Vina - Mean: {np.mean(opt_scores):.3f}, Median: {np.median(opt_scores):.3f}, Neg ratio: {opt_vina_results[\"neg_ratio\"]:.3f}')
        
        # QED (Quantitative Estimate of Drug-likeness)
        if qed_results['qeds']:
            qed_mean = np.mean(qed_results['qeds'])
            print(f'  QED: {qed_mean:.3f}')
        
        # PBR (PoseBuster Pass Ratio)
        pbr_ratio = pb_results['pbr'] / max(1, qed_results['protein_counter'] * 10) * 100
        print(f'  PBR: {pbr_ratio:.1f}%')
        
        # BNC (Better-than-Native Count)
        print(f'  BNC: {btn_results[\"btn_count\"]}')
        
        # Valid ratio
        if qed_results['protein_counter'] > 0:
            valid_ratio = len(qed_results['qeds']) / (qed_results['protein_counter'] * 10) * 100
            print(f'  Valid: {valid_ratio:.1f}%')
        
        # Interactions
        print(f'  Interactions: {interaction_results[\"interaction_count\"]}')
        
        # Strain energy
        if strain_results['strain_energy']:
            strain_mean = np.nanmean(strain_results['strain_energy'])
            print(f'  Strain Energy: {strain_mean:.3f}')
    else:
        print(f'\\n{name}: Files not found')

print('\\n=== Summary ===')
print('Best performing molecules:')
print('Vina scores (lower is better):')
best_vina_score = float('inf')
best_vina_name = ''

print('Molecular quality metrics:')
for name, data in results.items():
    if data['opt_vina']['scores']:
        vina_score = np.mean(data['opt_vina']['scores'])
        if vina_score < best_vina_score:
            best_vina_score = vina_score
            best_vina_name = name
        print(f'  {name}:')
        print(f'    Vina: {vina_score:.3f}')
        
        if data['qed']['qeds']:
            qed_mean = np.mean(data['qed']['qeds'])
            print(f'    QED: {qed_mean:.3f}')
        
        if data['interactions']['interaction_count']:
            print(f'    Interactions: {data[\"interactions\"][\"interaction_count\"]}')
        
        if data['strain_energy']['strain_energy']:
            strain_mean = np.nanmean(data['strain_energy']['strain_energy'])
            print(f'    Strain Energy: {strain_mean:.3f}')

if best_vina_name:
    print(f'\\nBest Vina: {best_vina_name} with mean score: {best_vina_score:.3f}')
"
else
    echo "Warning: generate_paper_tables.py not found. Results saved to files."
fi

echo ""
echo "=== Analysis Complete! ==="
echo "Results saved in subdirectories:"
for name in "${!FILES[@]}"; do
    echo "  - $name: ${OUTPUT_BASE_DIR}/${name}/"
done

echo ""
echo "To check individual results:"
echo "python scripts/check_vina_scores.py raw_mols/unconditional_noguidance_40atoms/"
echo "python scripts/check_vina_scores.py raw_mols/unconditional_guidance_40atoms/"
echo "python scripts/check_vina_scores.py raw_mols/conditional_guidance_30atoms/" 