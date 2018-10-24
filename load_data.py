import pandas as pd
import numpy as np
import pickle


def load_data(filepath, set_ids="all"):
    df_raw = pd.read_table(filepath, encoding="latin_1")
    print("Taille du set de base :",df_raw.shape)
    if set_ids == "all":
        df_raw = df_raw[['essay', "domain1_score"]]
        print("Taille du set :",df_raw.shape)
        df_raw.dropna()
        return df_raw.values
    else:
        dfs = []
        for i in set_ids:
            dfs.append(df_raw.loc[df_raw['essay_set'] == i])
        dfs = pd.concat(dfs)
        dfs = dfs[['essay', 'domain1_score']]
        print("Taille du set :",dfs.shape)
        df_raw.dropna()
        return dfs.values[:,0], dfs.values[:,-1:]

X_train, y_train = load_data("data/training_set.tsv", [5])

pickle.dump((X_train,y_train), open("data/data_set.pickle", "wb"))
