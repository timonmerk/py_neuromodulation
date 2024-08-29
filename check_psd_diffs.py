import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

    df = pd.read_csv(os.path.join(PATH_, "psds_all.csv"))
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
            if df_close.shape[0] != df_open.shape[0]:
                df_open = df_open.iloc[:df_close.shape[0]]
                
            arr_diff = df_close.iloc[:, 5:-2].values - df_open.iloc[:, 5:-2].values
            freqs = df_open.columns[5:-2]
            chs = df_open["ch"]
            l_mod.append(arr_diff)
        l_all.append(np.concatenate(l_mod, axis=0))
    
    flierprops = dict(marker='o', color='gray', markersize=1)
    plt.figure(figsize=(15, 9))
    for idx_mod, modality in enumerate(df["modality"].unique()):
        plt.subplot(2, 2, idx_mod+1)
        # limit range to 100 Hz
        idx_freq_below = np.where(freqs.astype(float) <= 100)[0]

        plt.boxplot(l_all[idx_mod][:, idx_freq_below][:, ::1], flierprops=flierprops, showfliers=False)
        # plot a horizontal line at 0
        plt.axhline(0, color="black", linestyle="--")
        # plot the mean of the differences
        plt.plot(np.median(l_all[idx_mod][:, idx_freq_below][:, ::1], axis=0), color="black", linewidth=2)
        plt.xticks(ticks=np.arange(1, len(freqs[idx_freq_below])+1)[::10][::1],
                   labels=freqs[idx_freq_below][::10][::1].astype(float).astype(int))
        plt.title(modality)
        plt.xlim(0, )
        plt.ylabel("Power difference")
        plt.xlabel("Frequency [Hz]")
    plt.suptitle("Power differences Eyes Close - Open")
    plt.tight_layout()
    plt.show(block=True)