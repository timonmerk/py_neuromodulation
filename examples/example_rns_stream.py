import os
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

def run_sub(sub_cohort):
    sub = sub_cohort[0]
    PATH_COHORT = sub_cohort[1]
    PATH_SUB = os.path.join(PATH_COHORT, sub)
    files = [os.path.join(PATH_SUB, f) for f in os.listdir(PATH_SUB) if '.dat' in f]
    # stream = init_pynm_rns_stream()
    stream = init_stream_all()
    PATH_OUT = os.path.join(PATH_OUT_BASE, cohort)
    try:
        stream.run(files, PATH_OUT, folder_name=sub)
    except:
        print(f'could not run {cohort} {sub}')

if __name__ == "__main__":

    PATH_BASE = 'Z:\\RNS_data\\ecog_reconstructed\\'
    PATH_OUT_BASE = r'X:\Users\timon\Ashley_pynm_datfiles\OUT_example_without_fooof'
    l_sub_cohort = []
    for cohort in ['MTS', 'MSR', 'PIT', 'MGH']:
        PATH_COHORT = os.path.join(PATH_BASE, cohort)
        subjects = [s for s in os.listdir(PATH_COHORT) if 'RNS' in s and os.path.exists(os.path.join(PATH_OUT_BASE, cohort, s)) is False]
        for sub in subjects:
            l_sub_cohort.append((sub, PATH_COHORT))
    #run_sub(l_sub_cohort[0])

    # multiprocessing
    #location = './cachedir'
    #memory = Memory(location, verbose=0)
    #costly_compute_cached = memory.cache(run_sub)
    #def data_processing_mean_using_cache(sub_cohort):
    #    """Compute the mean of a column."""
    #    return run_sub(sub_cohort)
    Parallel(n_jobs=17)(
        delayed(run_sub)(sub_cohort)
        for sub_cohort in l_sub_cohort
    )
    #memory.clear(warn=False)