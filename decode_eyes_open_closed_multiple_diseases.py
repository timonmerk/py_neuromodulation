import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
import numpy as np
from sklearn import metrics, model_selection, linear_model
from matplotlib.backends.backend_pdf import PdfPages


PATH_FILES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\Data\new data"

def balance_classes(df, target_column):
    min_class_size = df[target_column].value_counts().min()
    balanced_df = pd.concat([
        resample(df[df[target_column] == cls], 
                 replace=False, 
                 n_samples=min_class_size, 
                 random_state=42)
        for cls in df[target_column].unique()
    ])
    return balanced_df

if __name__ == "__main__":

    # get all folders in the PATH_FILES directory
    # save the confusion matrices as pdf
    pdf_path = os.path.join(PATH_FILES, "confusion_matrices_alpha_beta_gamma.pdf")
    pdf = PdfPages(pdf_path)

    folders = [f for f in os.listdir(PATH_FILES) if f.startswith("features_")]
    for sub in folders:
        df_orig = pd.read_csv(os.path.join(PATH_FILES, sub ))
        sub_name = sub[len("features_"):sub.find(".csv")]    
        subcortex_loc = "STN" if len([c for c in df_orig.columns if "STN" in c]) > 0 else "GPI" 

        ca_loc = []
        ba_loc = []

        np.random.seed()

        locs = [subcortex_loc, "ECOG", "EEG"]
        for loc in locs:
            df = df_orig.copy()
            features_bands = ["gamma", "alpha", "beta"]
            features_loc = [loc]
            
            # integer encode label column
            df["label_enc"] = df["label"].map({"SLEEP": 0, "EyesOpen": 1, "EyesClosed": 2})
            df = df.drop(columns=["time", "label", "disease", "subcortex", "subject"])

            # run a three fold cross validation to decode the label column, remove the time column
            cv = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
            model = linear_model.LinearRegression()
            cm_list = []
            ba_list = []
            for train_, test_ in cv.split(df, df["label_enc"]):
                # integer encode the label column
                train_df = df.iloc[train_]
                train_df_balanced = balance_classes(train_df, "label_enc")

                X_train = train_df_balanced.drop(columns=["label_enc"])
                y_train = train_df_balanced["label_enc"]

                # ensure that the same number of samples are taken for training from each class

                
                X_test = df.iloc[test_, :]
                y_test = df.iloc[test_, :]["label_enc"]
                X_test = X_test.drop(columns=["label_enc"])
                
                # include only columns that where all feature_list elements needs to be in the channel name
                #cols_use = [c for c in X_train.columns if all([f_band in c and f_loc in c for f_band, f_loc in zip(features_bands, features_loc)])]
                
                cols_use = []
                for c in X_train.columns:
                    for f_loc in features_loc:
                        for f_band in features_bands:
                            if f_loc in c and f_band in c:
                                cols_use.append(c)
                X_train = X_train[cols_use]
                X_test = X_test[cols_use]
                
                model = linear_model.LogisticRegression(class_weight="balanced")
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                # plot the confusion matrix
                cm = metrics.confusion_matrix(y_test, pred, normalize="true")
                #disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SLEEP", "EyesOpen", "EyesClosed"])
                #disp.plot(cmap=plt.cm.Blues)
                #plt.title("Confusion Matrix")
                #plt.show(block=True)

                cm_list.append(cm)
                ba_list.append(metrics.balanced_accuracy_score(y_test, pred))

                print(metrics.balanced_accuracy_score(y_test, pred))
            
            # plot the mean confusion matrix
            cm_mean = sum(cm_list) / len(cm_list)
            #disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=["SLEEP", "EyesOpen", "EyesClosed"])
            #disp.plot(cmap=plt.cm.Blues)
            #plt.title("Mean Confusion Matrix")
            #plt.show(block=True)

            ba_mean = sum(ba_list) / len(ba_list)
            ca_loc.append(cm_mean)
            ba_loc.append(np.round(ba_mean, 2))

        # make a subplot for each confusion matrix:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed
        for i, ax in enumerate(axs):
            disp = metrics.ConfusionMatrixDisplay(
                confusion_matrix=ca_loc[i],
                display_labels=["SLEEP", "EyesOpen", "EyesClosed"]
            )
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title(f"loc: {locs[i]} ba: {ba_loc[i]}")  # {df_orig['disease'].unique()[0]} sub: {sub_name}
        plt.suptitle(f"{df_orig['disease'].unique()[0]} sub: {sub_name}")
        plt.tight_layout()
        #plt.show(block=True)
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()