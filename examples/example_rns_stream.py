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

# Nexus2/RNS_DataBank/PITT/PIT-RNS1534/iEEG
# PIT-RNS1534_PE20161220-1_EOF_SZ-NZ.EDF


def init_pynm_rns_stream():
    # basic init settings that will initialize the stream
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
    stream.settings["features"]["fooof"] = False
    stream.settings["features"]["linelength"] = False
    stream.settings["features"]["sharpwave_analysis"] = False
    stream.settings["features"]["mne_connectiviy"] = False

    # SPECIFY specific feature modalities
    stream.settings["mne_connectiviy"]["method"] = "plv"
    stream.settings["mne_connectiviy"]["mode"] = "multitaper"

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
        [5, 30],
        [5, 60],
    ]

    stream.settings["fooof"]["periodic"]["center_frequency"] = False
    stream.settings["fooof"]["periodic"]["band_width"] = False
    stream.settings["fooof"]["periodic"]["height_over_ap"] = False
    stream.settings["fooof"]["windowlength_ms"] = 500000

    stream.init_stream(
        sfreq=sfreq,
        line_noise=60,
    )

    return stream


if __name__ == "__main__":

    PATH_BASE = r"C:\Users\ICN_admin\Documents\Datasets\RNS_Test\MGH"
    PATH_OUT = r"C:\Users\ICN_admin\Documents\Datasets\RNS_Test\OUT_Features"
    sub_name = "RNS2019"
    PATH_READ = os.path.join(PATH_BASE, sub_name)

    list_dat_file_paths = [
        os.path.join(PATH_READ, f) for f in os.listdir(PATH_READ)
    ]

    stream = init_pynm_rns_stream()
    stream.run(list_dat_file_paths, PATH_OUT, folder_name=sub_name)
