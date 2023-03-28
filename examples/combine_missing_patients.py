import pandas as pd
import os
import numpy as np
import shutil

PATH_DAT = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\AllReconstructedFiles\Features\remain"
PATH_OUT = r"C:\Users\ICN_admin\Documents\RNS_pynm_out\previous"#r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\AllReconstructedFiles\Features"

for cohort in ["MGH", "PIT", "MSR", "MTS"]:
    PATH_COHORT = os.path.join(PATH_DAT, cohort)
    ind_subs = np.unique([f[:7] for f in os.listdir(PATH_COHORT)])

    for sub in ind_subs:

        df_comb = []
        folders_sub = [f for f in os.listdir(PATH_COHORT) if sub in f]
        for folder_sub in folders_sub:
            file_read = [f for f in os.listdir(os.path.join(PATH_COHORT, folder_sub)) if "FEATURES" in f][0]
            df_comb.append(pd.read_csv(os.path.join(PATH_COHORT, folder_sub, file_read)))

        # save data
        df_tot = pd.concat(df_comb)
        PATH_SAVE_FOLDER = os.path.join(PATH_OUT, cohort, sub, f"{file_read[:7]}_{file_read[10:]}")
        PATH_OUT_FOLDER = os.path.join(PATH_OUT, cohort, sub)
        if os.path.exists(PATH_OUT_FOLDER) is False:
            os.mkdir(PATH_OUT_FOLDER)
        df_tot.to_csv(PATH_SAVE_FOLDER)

        # move also the remaining files in os.path.join(PATH_COHORT, folder_sub)
        # rename the files, remove the _x_

        for f in os.listdir(os.path.join(PATH_COHORT, folder_sub)):
            if "FEATURES" not in f:
                try:
                    shutil.move(
                        src=os.path.join(PATH_COHORT, folder_sub, f),
                        dst=os.path.join(PATH_OUT, cohort, sub, f"{f[:7]}_{f[10:]}")
                    )
                except Exception as e:
                    print(e)
