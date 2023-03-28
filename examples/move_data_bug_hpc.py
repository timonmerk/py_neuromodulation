import os
import shutil

PATH_DAT = r"C:\Users\ICN_admin\Documents\RNS_pynm_out\MGH"

PATH_ROOT = r"C:\Users\ICN_admin\Documents\RNS_pynm_out"

PATH_REFERENCE = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\AllReconstructedFiles"

# idea: check for each folder in PATH_DAT in which directory of PATH_REFERENCE it is located,
# then move it to the one in PATH_ROOT

subs = os.listdir(PATH_DAT)
cohorts = ["PIT", "MGH", "MSR", "MTS"]

for sub in subs:
    for cohort in cohorts:
        if sub in os.listdir(os.path.join(PATH_REFERENCE, cohort)):
            # move file to
            if os.path.exists(os.path.join(PATH_ROOT, cohort, sub)) is False:
                shutil.move(os.path.join(PATH_DAT, sub), os.path.join(PATH_ROOT, cohort))

# check which ones are missing
missing = []
for cohort in cohorts:
    for sub in os.listdir(os.path.join(PATH_REFERENCE, cohort)):
        if sub not in subs:
            missing.append((cohort, sub))

print(missing)

('PIT', 'RNS3016')
('PIT', 'RNS4098')
('PIT', 'RNS6762')
('PIT', 'RNS7168')
('PIT', 'RNS8326')
('MGH', 'RNS0351')
('MGH', 'RNS3569')
('MGH', 'RNS9357')
('MGH', 'RNS9674')
('MSR', 'RNS6204')
('MTS', 'RNS0834')
('MTS', 'RNS4554')
('MTS', 'RNS6222')

