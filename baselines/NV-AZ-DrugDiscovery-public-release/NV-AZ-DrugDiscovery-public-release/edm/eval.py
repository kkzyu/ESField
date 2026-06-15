import os
import pandas as pd
import numpy as np
import argparse


# function read all txt files in a folder
def read_txt_files(folder_path, vina_folder_name="mols_vina/"):
    mols_vina_path = os.path.join(folder_path, vina_folder_name)
    vina_list = []
    for file in os.listdir(mols_vina_path):
        if file.endswith(".txt") and file.startswith("ligand"):
            with open(os.path.join(mols_vina_path, file), "r") as score_file:
                line = score_file.readline()
                number_str = line.split(":")[1].split()[0]
                number = float(number_str)
                vina_list.append(number)

    # get mean, median, and 25% and 75% of the values of the vina_list
    result_mean = np.mean(vina_list)
    result_median = np.median(vina_list)
    result_one_quarter = np.percentile(vina_list, 25)
    result_three_quarters = np.percentile(vina_list, 75)
    result_min = min(vina_list)
    result_max = max(vina_list)
    results = [
        result_mean,
        result_median,
        result_one_quarter,
        result_three_quarters,
        result_min,
        result_max,
    ]
    return results


if __name__ == "__main__":

    # get the folder paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=".")
    parser.add_argument("--vina_folder_name", type=str, default="mols_vina/")
    args = parser.parse_args()
    # loop through all folders in the data folder
    df = pd.DataFrame()
    # df.columns = ["mean", "median", "1quarter", "3quarters", "min", "max"]

    for folder in os.listdir(args.file_path):
        if os.path.isdir(os.path.join(args.file_path, folder)):
            print(folder)
            results = read_txt_files(
                os.path.join(args.file_path, folder), args.vina_folder_name
            )
            # print (results)
            # concat the results to the dataframe row name is the folder name
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [results],
                        columns=[
                            "mean",
                            "median",
                            "1quarter",
                            "3quarters",
                            "min",
                            "max",
                        ],
                        index=[folder],
                    ),
                ]
            )

    # sort the dataframe by the row names
    df = df.sort_index()
    # make index a column named start_t
    df = df.reset_index()

    print(df)
    df.to_csv(
        os.path.join(args.file_path, f"{args.vina_folder_name}_results.csv"),
        index=False,
    )
