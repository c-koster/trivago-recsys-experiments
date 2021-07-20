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
for i in range(1,6):
    print("n_users with more than {} session(s): {}".format(i,counts[counts.sessions > i].shape[0]))
    v = 100*i - 1
    print("n_users with more than {} interactions: {}\n".format(v,counts[counts.interactions > v].shape[0]))
