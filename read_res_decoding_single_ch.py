import pandas as pd
import os

from matplotlib import pyplot as plt
import seaborn as sns

PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

mods = ["alpha", "fft"]

l_ = []
for mod in mods:
    df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_{mod}.csv"))
    df["mod"] = mod
    l_.append(df)

df_all = pd.concat(l_)

plt.figure()
sns.boxplot(data=df_all, x="loc", y="ba", hue="mod", palette="viridis")
plt.tight_layout()
plt.show(block=True)

df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_fft.csv"))
# melt the dataframe that all columns with coef_ become a column
df_melt = df.melt(id_vars=["ba", "loc", "sub", "dout"], value_vars=[c for c in df.columns if "coef_" in c])

plt.figure()
# melt the dataframe that coef_ becomes a column
sns.boxplot(data=df_melt, x="variable", y="value", palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("Coef")
plt.tight_layout()
plt.show(block=True)