import numpy as np


def training_weights(df_train, toxicity_column, identity_columns):
    subgroup_positive = (df_train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)
    subgroup_negative = (df_train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)

    background_positive = (df_train[toxicity_column].values >= 0.5).astype(bool).astype(np.int)
    background_negative = (df_train[toxicity_column].values < 0.5).astype(bool).astype(np.int)

    weights = np.ones((len(df_train),)) / 4
    weights += (df_train[identity_columns].fillna(0).values >= 0.5).mean(axis=1) / 4
    weights += ((background_positive + subgroup_negative) > 1).astype(bool).astype(np.int) / 4
    weights += ((background_negative + subgroup_positive) > 1).astype(bool).astype(np.int) / 4
    return weights


def training_weights_s(df_train, toxicity_column, identity_columns):
    weights = np.ones((len(df_train),))
    weights += df_train[identity_columns].fillna(0).values.sum(axis=1) * 3
    weights += df_train[toxicity_column].values * 8
    weights /= weights.max()
    return weights
