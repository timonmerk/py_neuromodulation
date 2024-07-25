import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
import numpy as np
from sklearn import metrics, model_selection, linear_model

PATH_FEATURES = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\EyesOpenBeijing\Data\features_all.csv"
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

    ca_loc = []
    ba_loc = []

    np.random.seed()

    locs = ["STN", "ECOG", "EEG"]
    for loc in locs:
        features = ["alpha",]
        if loc == "STN":
            features = ["alpha", "LSTN1-LSTN2", "RSTN3-RSTN3"]
        features = features + [loc]

        df = pd.read_csv(PATH_FEATURES)
        
        # integer encode label column
        df["label_enc"] = df["label"].map({"SLEEP": 0, "EyesOpen": 1, "EyesClosed": 2})
        df = df.drop(columns=["time", "label"])

        # run a three fold cross validation to decode the label column, remove the time column
        cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
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
            cols_use = [c for c in X_train.columns if all([f in c for f in features])]
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
        ax.set_title(f"loc: {locs[i]} Mean Confusion Matrix {ba_loc[i]}")
    plt.tight_layout()
    plt.show(block=True)