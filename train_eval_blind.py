"""
This will look suspiciously like train eval. Some DIFFERENCES include:
    1. remove 'is_avantaged_user' as a feature but still evaluate each group individually
    2.

"""
import pandas as pd


data_all = pd.read_parquet("data/trivago/data_all_user_blind.parquet", engine="pyarrow")

feature_names = set(data_all.columns) - set(["y", "q_id", "grp", "choice_idx","is_advantaged_user"])

# split the data
train = data_all[data_all.grp == 0]
vali = data_all[data_all.grp == 1]
test = data_all[data_all.grp == 2]
