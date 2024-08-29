import mne
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fooof import FOOOF

def get_fooof_params(freqs, data_fit, PLT_FOOOF=False):
    fm = FOOOF(aperiodic_mode="fixed", max_n_peaks=4)
    
    
    freqs_blow_100 = np.where(freqs <= 100)[0]
    freqs = freqs[freqs_blow_100]
    data_fit = data_fit[freqs_blow_100]

    fm.fit(freqs, data_fit, freq_range=[40, 100])
    offset = fm.get_params("aperiodic_params", col="offset")
    exponent = fm.get_params("aperiodic_params", col="exponent")
    if PLT_FOOOF:
        fm.plot(plt_log=True, log_freqs=True)
        plt.show(block=True)
    return offset, exponent


if __name__ == "__main__":
    
    PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

    files_all = [f for f in os.listdir(PATH_) if f.endswith(".fif")]
    subs = np.unique([f.split("_")[2] for f in files_all])
    df_ = pd.DataFrame()
    l_ = []
    PLT_ = False
    CALC_SPEC = True

    pdf_path = os.path.join(PATH_, "psds_name__.pdf")
    with PdfPages(pdf_path) as pdf:
        for sub in subs[:]:
            files_sub = [f for f in files_all if sub in f]
            conditions_ = [f.split("_")[3] for f in files_sub]
            disease = files_sub[0].split("_")[4]
            psds_ = []
            print(sub)

            for f in files_sub:
                raw = mne.io.read_raw_fif(os.path.join(PATH_, f))
                psd_ = raw.compute_psd()
                freqs = psd_.freqs
                power = np.log(psd_.get_data())
                ch_names = raw.ch_names
                psds_.append(power)
                idx_below_100 = np.where(freqs <= 500)[0]

                if CALC_SPEC:
                    for ch_idx, ch in enumerate(ch_names):
                        d_ = {}
                        d_["sub"] = str(sub)
                        d_["disease"] = disease
                        d_["f_name"] = f
                        d_["ch"] = ch
                        offset, exponent = get_fooof_params(freqs, np.exp(power[ch_idx, :]))
                        d_["offset"] = offset
                        d_["exponent"] = exponent

                        for freq in freqs:
                            d_[float(freq)] = float(power[ch_idx, np.where(freqs == freq)[0]])

                        l_.append(d_)
            channels = raw.ch_names

            if PLT_ is False:
                continue
            fig = plt.figure(figsize=(15, 15))
            for ch_idx, ch in enumerate(channels):
                print(ch)
                #if "bad" in ch:
                #    continue
                num_rows = len(channels) // 3 + 1
                plt.subplot(num_rows, 3, ch_idx+1)
                for i in range(len(psds_)):
                    if ch_idx == 0:
                        plt.plot(freqs[idx_below_100], psds_[i][ch_idx, idx_below_100], label=conditions_[i])
                    else: 
                        plt.plot(freqs[idx_below_100], psds_[i][ch_idx, idx_below_100])
                if ch_idx == 0:
                    plt.legend()
                #plt.xlim(0, 100)
                plt.title(ch)
                plt.xscale("log")
            #plt.show(block=True)
            plt.suptitle(f"{disease} {sub}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    if CALC_SPEC:
        df_ = pd.DataFrame(l_)
        df_.to_csv(os.path.join(PATH_, "psds_all_fitting_range_40_100.csv"))