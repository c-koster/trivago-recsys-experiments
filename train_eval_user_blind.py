"""
We think that a <> performs best on this dataset, so we run a parallel experiment (using <> models)
as follows.

1. extract features for the train_ and test_hashed datasets, and put these into a parquet file.

2. Tune models on this data and evaluate on the test set.


"""


# first i would like to find out how many 'advantaged' vs 'disadvantaged' users
# there are. this means users with d > 100 interactions (or try some other way to slice)
import pandas as pd

df = pd.read_csv("data/trivago/train.csv")

interactions_series = df["user_id"].value_counts()
interactions = interactions_series.to_frame(name="interactions")


sessions_series = df.groupby("user_id")["session_id"].nunique()
sessions = sessions_series.to_frame(name="sessions")

# full outer join --
counts = pd.merge(sessions,interactions,how='outer',left_index=True,right_index=True)


print("n_users in dataset {}".format(counts.shape[0]))
print("n_users with more than two sessions: {}".format(counts[counts.sessions > 2].shape[0]))
print("n_users with more than 99 interactions: {}".format(counts[counts.interactions > 99].shape[0]))