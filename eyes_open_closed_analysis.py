PATH_FILES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

from matplotlib import pyplot as plt
import py_neuromodulation as nm
from py_neuromodulation import nm_analysis, nm_define_nmchannels, nm_plots, NMSettings

import mne
import os
import pandas as pd
from joblib import Parallel, delayed


def infer_ch_types(ch_names):
    ch_types = []
    for ch in ch_names:
        if "STN" in ch or "GPI" in ch:
            ch_types.append("dbs")
        elif "ECOG" in ch:
            ch_types.append("ecog")
        elif "EEG" in ch:
            ch_types.append("eeg")
        else:
            ch_types.append("misc")
    return ch_types

def compute_subject(f):
    raw = mne.io.read_raw_fif(os.path.join(PATH_FILES, f))
    if "sleep" in f:
        label = "SLEEP"
    elif "open" in f:
        label = "EyesOpen"
    elif "close" in f:
        label = "EyesClosed" 
    data = raw.get_data()
    #raw.plot(block=True)
    #raw.compute_psd().plot()
    #plt.show(block=True)

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=infer_ch_types(raw.ch_names),
        used_types=["ecog", "dbs", "eeg"],
    )

    settings = NMSettings.get_fast_compute()
    settings.preprocessing = ["raw_resampling", "notch_filter"]
    settings.postprocessing["feature_normalization"] = False

    #settings.features.fooof = True

    stream = nm.Stream(
        settings=settings,
        nm_channels=nm_channels,
        verbose=True,
        sfreq=raw.info["sfreq"],
        line_noise=50,
        sampling_rate_features_hz=1,
    )

    features = stream.run(
        data,
        out_path_root=os.path.join(PATH_FILES, "features"),
        folder_name=f[:-4],)
    features["label"] = label
    features["sub"] = f.split("_")[2]
    features["disease"] = f.split("_")[4]
    features["file_name"] = f

    return features

if __name__ == "__main__":

    feature_l = []

    files_ = [ f for f in os.listdir(PATH_FILES) if ".fif" in f]

    compute_subject(files_[0])
    # use joblib to parallelize the computation and concatenate the output
    feature_l = Parallel(n_jobs=len(files_))(delayed(compute_subject)(f) for f in files_)
    
    df_all = pd.concat(feature_l)
    df_all.to_csv(os.path.join(PATH_FILES, "features_all.csv"), index=False)
    df_all.to_csv("features_all.csv", index=False)

    
