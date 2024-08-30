import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib import pyplot as plt
from fooof import FOOOF

if __name__ == "__main__":

    PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

    df = pd.read_csv(os.path.join(PATH_, "psds_all.csv"))
    # remove offset and exponent columns
    df = df.drop(columns=["offset", "exponent"])

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
    

    def get_diff_disease(df):
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

                arr_diff = df_close.iloc[:, 5:-2].values - df_open.iloc[:, 5:-2].values
                freqs = df_open.columns[5:-2]
                chs = df_open["ch"]
                l_mod.append(arr_diff)
            l_all.append(np.concatenate(l_mod, axis=0))
        return l_all, freqs
    

    PLT_BOXPLT = False


    l_all_ = []
    # get four colors from viridis

    diseases_ = df["disease"].unique()
    colors = sns.color_palette("viridis", len(diseases_))
    #for disease in diseases_:
        #df_disease = df.query(f"disease == '{disease}'").copy()
        #l_all, freqs = get_diff_disease(df_disease.copy())
        #l_all_.append(l_all)

    flierprops = dict(marker='o', color='gray', markersize=1)
    plt.figure(figsize=(15, 9))
    for idx_mod, modality in enumerate(df["modality"].unique()):
        plt.subplot(2, 2, idx_mod + 1)
        

        for disease_idx, disease in enumerate(diseases_):
            df_disease = df.query(f"disease == '{disease}' and modality == '{modality}'").copy()
            if df_disease.shape[0] == 0:
                continue
            l_all, freqs = get_diff_disease(df_disease.copy())
            idx_freq_below = np.where(freqs.astype(float) <= 100)[0]
            median_diff = np.median(l_all[0][:, idx_freq_below][:, ::1], axis=0)
            std_diff = np.std(l_all[0][:, idx_freq_below][:, ::1], axis=0)
            plt.plot(median_diff, linewidth=2, color=colors[disease_idx], label=disease)
            #plt.fill_between(np.arange(len(median_diff)), median_diff - std_diff, median_diff + std_diff,
            #                 color=colors[disease_idx],
            #                 alpha=0.5
            #)
            
        plt.axhline(0, color="black", linestyle="--")
        plt.xticks(ticks=np.arange(1, len(freqs[idx_freq_below]) + 1)[::10],
                labels=freqs[idx_freq_below][::10].astype(float).astype(int))

        plt.title(modality)
        plt.legend()
        plt.xlim(0, len(median_diff))
        plt.ylabel("Power difference")
        plt.xlabel("Frequency [Hz]")

    plt.suptitle("Power differences Eyes Close - Open")
    plt.tight_layout()
    plt.show(block=True)