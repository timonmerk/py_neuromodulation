import pandas as pd
import os

from matplotlib import pyplot as plt
import seaborn as sns

PATH_ = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\2708"

mods = ["alpha", "fft"]

l_ = []
for mod in mods:
    df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_{mod}_three_class.csv"))
    df["mod"] = mod
    l_.append(df)

df_all = pd.concat(l_)

plt.figure(figsize=(15, 12))
diseases_ = df_all.dout.unique()
locs_ = df_all["loc"].unique()

for disease_idx, disease in enumerate(diseases_):
    for loc_idx, loc_ in enumerate(locs_):
        df_disease = df_all[df_all["dout"] == disease]
        df_disease = df_disease[df_disease["loc"] == loc_]
        plt.subplot(len(diseases_), len(locs_) , len(locs_)*disease_idx + 1 + loc_idx)
        sns.boxplot(data=df_disease, x="mod", y="ba", palette="viridis")
        plt.title(f"{loc_} {disease}")
        #df_disease = df_all[df_all["dout"] == disease]
        #plt.subplot(1, len(diseases_), disease_idx + 1)
        #sns.boxplot(data=df_all, x="loc", y="ba", hue="mod", palette="viridis")
plt.tight_layout()
plt.show(block=True)

df = pd.read_csv(os.path.join(PATH_, f"out_per_loc_mod_fft.csv"))
# melt the dataframe that all columns with coef_ become a column
df_melt = df.melt(id_vars=["ba", "loc", "sub", "dout"], value_vars=[c for c in df.columns if "coef_" in c])

plt.figure(figsize=(15, 12))
# melt the dataframe that coef_ becomes a column
sns.boxplot(data=df_melt, x="variable", y="value", palette="viridis")
plt.xticks(rotation=90)
plt.ylabel("Coef")
plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(15, 12))
for disease_idx, disease in enumerate(diseases_):
    for loc_idx, loc_ in enumerate(locs_):
        df_disease = df_melt[df_melt["dout"] == disease]
        df_disease = df_disease[df_disease["loc"] == loc_]
        plt.subplot(len(diseases_), len(locs_) , len(locs_)*disease_idx + 1 + loc_idx)
        sns.boxplot(data=df_disease, x="variable", y="value", palette="viridis")
        plt.title(f"{loc_} {disease}")
        #df_disease = df_all[df_all["dout"] == disease]
        #plt.subplot(1, len(diseases_), disease_idx + 1)
        #sns.boxplot(data=df_all, x="loc", y="ba", hue="mod", palette="viridis")
plt.tight_layout()
plt.show(block=True)


#  .groupby(["sub"]).max()

plt.figure(figsize=(4, 3), dpi=300)
sns.boxplot(data=df.query("loc == 'STN'"), x="dout", y="ba")
sns.swarmplot(data=df.query("loc == 'STN'"), x="dout", y="ba", color="gray", alpha=0.5)
plt.title("STN sleep-eyes open-eye closed alpha only")
plt.tight_layout()
plt.show(block=True)