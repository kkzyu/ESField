import os
import glob
import pandas as pd

# Base directory
base_dir = "evaluation/with_results"
merged_df = pd.DataFrame()  # Create an empty DataFrame to store the merged results

# Loop through each subdirectory in the base directory
for folder in glob.glob(os.path.join(base_dir, "*/")):
    folder_name = os.path.basename(os.path.normpath(folder))  # Get the folder name

    # Look for the "mols_vina_results" file in the folder
    result_file = os.path.join(folder, "mols_vina_results.csv")

    # Check if the file exists
    if os.path.exists(result_file):
        try:
            # Read the file into a DataFrame
            df = pd.read_csv(result_file)

            # Add the folder name as a new column
            df["Folder"] = folder_name
            # df['start_t'] = df.index
            # Concatenate the current DataFrame to the merged DataFrame along the columns (axis=1)
            merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

            print(f"Processed file in {folder_name}")
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    else:
        print(f"No mols_vina_results found in {folder_name}")

# change the 'index' column to 'start_t'
merged_df = merged_df.rename(columns={"index": "start_t"})

# analyze the merged_df
# for each folder find the start_t of each lowest mean value

mean_stats = merged_df["start_t"][merged_df.groupby("Folder")["mean"].idxmin()]
print("mean value count", mean_stats.value_counts())

min_stats = merged_df["start_t"][merged_df.groupby("Folder")["min"].idxmin()]
print("min value count", min_stats.value_counts())

median_stats = merged_df["start_t"][merged_df.groupby("Folder")["median"].idxmin()]
print("median value count", median_stats.value_counts())

quarter_stats = merged_df["start_t"][merged_df.groupby("Folder")["1quarter"].idxmin()]
print(" 25% value count", quarter_stats.value_counts())

three_quarter_stats = merged_df["start_t"][
    merged_df.groupby("Folder")["3quarters"].idxmin()
]
print("75% value count", three_quarter_stats.value_counts())


# in the merged_df, for each folder find with index column the lowest mean value, plot the histogram of the index column
# and save the plot as a png file


# merged_df.to_csv("merged_mols_vina_results.csv", index=False)
# print("Merged results saved as merged_mols_vina_results.csv")
