PATH_FILES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\Data\new data"

from matplotlib import pyplot as plt
import py_neuromodulation as nm
from py_neuromodulation import nm_analysis, nm_define_nmchannels, nm_plots, NMSettings

import mne
import os
import pandas as pd
from joblib import Parallel, delayed

def compute_subject(f):
    raw = mne.io.read_raw_fif(os.path.join(PATH_FILES, f))
    if "sleep" in f:
        label = "SLEEP"
    elif "open" in f:  # what means opclo
        label = "EyesOpen"
    elif "close" in f:
        label = "EyesClosed" 
    data = raw.get_data()

    sub = f[f.find("_Bi_")+4:f.find("_Bi_")+4+5]
    disease = "PD" if "_PD_" in f else "Meige" if "_Meige_" in f else "None"
    subcortex = "STN" if "STN" in f else "GPi" if "GPI" in f else "None"

    file_name = f"{sub}_{disease}_{subcortex}_{label}"

    #raw.plot(block=True)
    #raw.compute_psd().plot()
    #plt.show(block=True)

    ch_names = raw.ch_names
    print(raw.ch_names)
    # set ch types to be dbs if STN or GPI is in the name, ecog if ECOG is in the name, and eeg if EEG is in the name else misc
    ch_types = ["dbs" if "STN" in ch or "GPI" in ch else "ecog" if "ECOG" in ch else "eeg" if "EEG" in ch else "misc" for ch in ch_names]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=ch_types,
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

    features = stream.run(data, out_path_root=PATH_FILES, folder_name=file_name)
    features["label"] = label
    features["subject"] = sub
    features["disease"] = disease
    features["subcortex"] = subcortex

    return features

if __name__ == "__main__":

    subjects = ["008XH", "009ZL", "023LA", "032ZL"]

    for sub in subjects:
        files_ = [ f for f in os.listdir(PATH_FILES) if ".fif" in f and sub in f]
        feature_l = []
        for f in files_:
            feature = compute_subject(f)
            feature_l.append(feature)
            # Note: I changed for P_Bi_008XHM_open_Meige_GPIEEGECOG_noise_1000Hz to open from openclosed
        df_sub = pd.concat(feature_l)
        df_sub.to_csv(os.path.join(PATH_FILES, f"features_{sub}.csv"), index=False)

        # use joblib to parallelize the computation and concatenate the output
        #feature_l = Parallel(n_jobs=len(files_))(delayed(compute_subject)(f) for f in files_)
        
        #df_all = pd.concat(feature_l)
        #df_all.to_csv(os.path.join(PATH_FILES, "features_all_multiple_diseases.csv"), index=False)

        
