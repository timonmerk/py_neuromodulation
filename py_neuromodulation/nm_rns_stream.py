import os
import numpy as np
import pandas as pd

from py_neuromodulation import nm_stream_abc, nm_oscillatory, nm_features


class Stream(nm_stream_abc.PNStream):
    def read_data(self, PATH_DAT):
        b = open(PATH_DAT, "rb").read()
        ecog = np.frombuffer(b, dtype=np.int16)
        ecog = ecog - 512
        ecog = ecog.reshape([-1, 4])
        return ecog.T

    def run(
        self,
        list_dat_file_paths: list,
        out_path_root: str,
        folder_name: str = "sub",
    ):

        features = []

        for dat_file in list_dat_file_paths:
            data_dat_file = self.read_data(dat_file)
            fft_features_pre = self.features.features[0]
            s = self.settings
            s["fft_settings"]["windowlength_ms"] = (
                data_dat_file.shape[1] * 4
            )  # 4 ms
            self.run_analysis.features = nm_features.Features(
                s, fft_features_pre.ch_names, fft_features_pre.sfreq
            )

            f = self.run_analysis.process_data(data_dat_file)
            f["dat_file_number"] = os.path.basename(dat_file)[:-4]
            features.append(f)

        features_df = pd.DataFrame(features)
        self.save_after_stream(out_path_root, folder_name, features_df)

    def _add_timestamp(
        self, feature_series: pd.Series, idx: int | None = None
    ) -> pd.Series:
        return feature_series
