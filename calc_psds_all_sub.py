import mne
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



if __name__ == "__main__":
    
    PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

    files_all = [f for f in os.listdir(PATH_) if f.endswith(".fif")]
    subs = np.unique([f.split("_")[2] for f in files_all])

    pdf_path = os.path.join(PATH_, "psds_all.pdf")
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
                idx_below_100 = np.where(freqs <= 100)[0]
            
            channels = raw.ch_names

            fig = plt.figure(figsize=(15, 15))
            for ch_idx, ch in enumerate(channels):
                print(ch)
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
            #plt.show(block=True)
            plt.suptitle(f"{disease} {sub}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
