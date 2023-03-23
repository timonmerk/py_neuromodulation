import os
import sys
import py_neuromodulation as nm
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_stream_offline,
    nm_rns_stream,
    nm_IO,
    nm_plots,
    nm_stats,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import mne
from scipy import stats
import pandas as pd
from joblib import Memory
from joblib import Parallel, delayed

# Nexus2/RNS_DataBank/PITT/PIT-RNS1534/iEEG
# PIT-RNS1534_PE20161220-1_EOF_SZ-NZ.EDF

def init_stream_all():
    channels = ["ch1", "ch2", "ch3", "ch4"]
    sfreq = 250

    ch_names = list(channels)
    ch_types = ["ecog" for _ in range(len(ch_names))]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference=None,
        bads=None,
        new_names="default",
        used_types=["ecog"],
    )

    stream = nm_rns_stream.Stream(
        settings=None,
        nm_channels=nm_channels,
        verbose=True,  # Change here if you want to see the outputs of the run
    )

    stream.reset_settings()

    # note: I limit here the frequency bands to have reliable data up to 60 Hz
    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [35, 60],
        "broadband" : [20, 120]
    }

    # INIT Feature Estimation Time Window Length and Frequency
    stream.settings[
        "sampling_rate_features_hz"
    ] = 0.1  # this should be obsolete now
    stream.settings["segment_length_features_ms"] = (
        500 * 1000
    )  # select here the maximum time range

    # ENABLE feature types
    stream.settings["features"]["fft"] = True
    stream.settings["features"]["linelength"] = True
    stream.settings["features"]["sharpwave_analysis"] = True
    stream.settings["features"]["bursts"] = True
    stream.settings["features"]["mne_connectiviy"] = False
    stream.settings["features"]["fooof"] = True
    stream.settings["fooof"]["periodic"]["center_frequency"] = False
    stream.settings["fooof"]["periodic"]["height_over_ap"] = False
    stream.settings["fooof"]["periodic"]["band_width"] = False
    stream.settings["fooof"]["windowlength_ms"] = 500000

    stream.settings["features"]["coherence"] = True
    stream.settings["coherence"]["channels"] = [
        [
            "ch1",
            "ch2"
        ],  
        [
            "ch1",
            "ch3"
        ],
        [
            "ch3",
            "ch4"
        ],
    ]
    stream.settings["coherence"]["frequency_bands"] = [
        "high beta",
        "low gamma"
    ]

    stream.settings["fft_settings"]["windowlength_ms"] = 500000
    stream.settings["fft_settings"]["log_transform"] = True
    stream.settings["fft_settings"]["kalman_filter"] = False

    sharpwave_settings_enable = [
        "width",
        "interval",
        "decay_time",
        "rise_time",
        "rise_steepness",
        "decay_steepness",
        "prominence",
        "interval",
        "sharpness",
    ]

    sharpwave_settings_disable = ["peak_left", "peak_right", "trough"]

    for f in sharpwave_settings_disable:
        stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
            f
        ] = False

    for f in sharpwave_settings_enable:
        stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
            f
        ] = True

    # For the distribution of sharpwave features in the interval (e.g. 10s) an estimator need to be defined
    # e.g. mean or max
    stream.settings["sharpwave_analysis_settings"]["estimator"][
        "mean"
    ] = sharpwave_settings_enable

    stream.settings["sharpwave_analysis_settings"]["estimator"]["max"] = [
        "sharpness",
        "prominence",
    ]
    stream.settings["sharpwave_analysis_settings"]["filter_ranges_hz"] = [
        [5, 30],[5, 60]
    ]

    stream.init_stream(
        sfreq=sfreq,
        line_noise=60,
    )
    return stream

def run_sub(f):
    files = f[1]
    file_idx = f[0]
    #files = files[:5]
    # stream = init_pynm_rns_stream()
    stream = init_stream_all()
    PATH_OUT = os.path.join(PATH_OUT_BASE, cohort)
    try:
        stream.run(files, PATH_OUT, folder_name=f"{sub}_{file_idx}")
    except:
        print(f'could not run {cohort} {sub}')

if __name__ == "__main__":

    #run_idx = int(sys.argv[1])
    run_idx = 0

    #PATH_BASE = "/data/gpfs-1/users/merkt_c/work/Data_RNS"
    #PATH_OUT_BASE = "/data/gpfs-1/users/merkt_c/work/OUT"

    PATH_BASE = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\AllReconstructedFiles\MTS"
    PATH_OUT_BASE = r"C:\Users\ICN_admin\Downloads\OUT\remain"

    #l_sub_cohort = []
    #for cohort in ['MTS', 'MSR', 'PIT', 'MGH']:
    #    PATH_COHORT = os.path.join(PATH_BASE, cohort)
    #    subjects = [s for s in os.listdir(PATH_COHORT) if 'RNS' in s and os.path.exists(os.path.join(PATH_OUT_BASE, cohort, s)) is False]
    #    for sub in subjects:
    #        l_sub_cohort.append((sub, PATH_COHORT))

    #run_sub(l_sub_cohort[run_idx])

    l_sub_cohort = [
        ('PIT', 'RNS3016'),
        ('PIT', 'RNS4098'),
        ('PIT', 'RNS6762'),
        ('PIT', 'RNS7168'),
        ('PIT', 'RNS8326'),
        ('MGH', 'RNS0351'),
        ('MGH', 'RNS3569'),
        ('MGH', 'RNS9357'),
        ('MGH', 'RNS9674'),
        ('MSR', 'RNS6204'),
        ('MTS', 'RNS0834'),
        ('MTS', 'RNS4554'),
        ('MTS', 'RNS6222'),
    ]

    cohort, sub = l_sub_cohort[run_idx]

    PATH_SUB = os.path.join(PATH_BASE, cohort, sub)

    # split the files into 12
    files = [os.path.join(PATH_SUB, f) for f in os.listdir(PATH_SUB) if '.dat' in f]

    files_group = np.array_split(files, 12)

    run_sub((0, files_group[0]))
    Parallel(n_jobs=12)(
        delayed(run_sub)((file_idx, file_group))
        for file_idx, file_group in enumerate(files_group)
    )
