#!/bin/bash

target_dir="evaluation/with_results/"

# List all folders in the current directory
for folder in "$target_dir"*/; do
    # Remove trailing slash from folder name
    folder_name="${folder%/}"
    
    # Run the Python script and pass the folder name as an argument
    python eval.py --file_path "$folder_name" --vina_folder_name mols_vina
    
    echo "Processed folder: $folder_name"
done
