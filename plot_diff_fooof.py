import seaborn as sns
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":

    PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

    df = pd.read_csv(os.path.join(PATH_, "psds_all_fitting_range_40_100.csv"))
    def get_modality(row):
        if "ECOG" in row["ch"]:
            return "ECOG"
        elif "EEG" in row["ch"]:
            return "EEG"
        elif "STN" in row["ch"]:
            return "STN"
        elif "GPI" in row["ch"]:
            return "GPi"
        else:
            return "Unknown"

    df["modality"] = df[["ch"]].apply(get_modality, axis=1)
    df["condition"] = df["f_name"].apply(lambda x: x.split("_")[3])

    # replace condition "open2" with "open"
    df["condition"] = df["condition"].replace("open2", "open")
    df["condition"] = df["condition"].replace("sleepns", "sleep")
    

    l_all = []
    for modality in df["modality"].unique():
        l_mod = []
        for sub in df["sub"].unique():    
            df_open = df[(df["sub"] == sub) & (df["modality"] == modality) & (df["condition"] == "open")]
            df_close = df[(df["sub"] == sub) & (df["modality"] == modality) & (df["condition"] == "close")]
            # remove channels that contain "bad"
            df_open = df_open[~df_open["ch"].str.contains("bad")]
            df_close = df_close[~df_close["ch"].str.contains("bad")]
            if df_close.shape[0] != df_open.shape[0]:
                df_open = df_open.iloc[:df_close.shape[0]]

            arr_diff_offset = df_close["offset"].values - df_open["offset"].values
            arr_diff_exponent = df_close["exponent"].values - df_open["exponent"].values

            chs = df_open["ch"]
            l_mod.append(np.concat([arr_diff_offset[:, np.newaxis], arr_diff_exponent[:, np.newaxis]], axis=1))
        l_all.append(np.concatenate(l_mod, axis=0))
    

    plt.figure()
    for idx_mod, modality in enumerate(df["modality"].unique()):
        plt.subplot(4, 1, idx_mod+1)
        sns.boxplot(data=l_all[idx_mod], showfliers=False, showmeans=True)
        # show individual data points
        sns.swarmplot(data=l_all[idx_mod], color=".25")
        plt.xticks(np.arange(2), ["offset", "exponent"])
        plt.title(modality)
    plt.tight_layout()
    plt.show(block=True)
    
    # I want to show the upper plot but as a histogram with both groups overlayed

    plt.figure()
    
    for idx_plt in range(2):
        for idx_mod, modality in enumerate(df["modality"].unique()):
            plt.subplot(4, 2, 4*idx_plt+1+idx_mod)
            sns.histplot(data=l_all[idx_mod][:, idx_plt], bins=50, kde=True)
            str_title = f"{modality} {['offset', 'exponent'][idx_plt]}"
            plt.title(str_title)
    plt.suptitle("closed - open")
    plt.tight_layout()
    plt.show(block=True)
