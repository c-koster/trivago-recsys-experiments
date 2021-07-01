"""
Compute click-through ratios for hotels. Do this two ways.

First, by looping over the clickouts in the training set and computing for each hotel i:
    ctr_i = (times hotel i is clicked)/(# times hotel i appears in impression results).

Second -- intuitively, we'd like the click through ratio to be penalized most when the
first-rank item is not clicked and rewarded most when it is clicked at a higher rank.
"""
from typing import Dict, List, Tuple

import pandas as pd
from collections import defaultdict, Counter


compute_ctr: Dict = defaultdict(lambda: {"num":0, "num_pw":0, "denom":0, "denom_pw": 0})
rank_click_prob: Dict[int,float] = {}

# helpers for main function -- these get applied to df rows
def create_entry(interaction_row: Tuple[str,int]) -> None:
    # unpack tuple. (item_id, rank_appeared)
    item_id = interaction_row[0]
    rank_appeared = interaction_row[1]
    entry = compute_ctr[item_id]
    entry["num"] += 1
    entry["num_pw"] += rank_click_prob[rank_appeared]


def count_denom(L: List) -> None:
    for idx, id in enumerate(L):
        compute_ctr[id]["denom"] +=1
        compute_ctr[id]["denom_pw"] += rank_click_prob[idx + 1] # rank = index + 1


def main() -> None:
    df = pd.read_csv("data/trivago/train.csv")
    df = df[df.action_type == "clickout item"]
    df["impressions"] = df["impressions"].str.split("|").map(list)


    # meanwhile: get the index of the reference in the impression. Adding one makes this the rank.
    get_rank = lambda x: 1 + x[1].index(x[0]) if x[0] in x[1] else 0
    df["rank_appeared"] = df[["reference","impressions"]].apply(get_rank,axis=1)
    # apply this over ref and impressions (x[0] is reference, x[1] is impressions)
    df = df[df.rank_appeared != 0]  # if the reference is not in the impressions list -- remove it


    #import matplotlib.pyplot as plt
    #plt.plot(df["rank_appeared"].value_counts().sort_index())
    #plt.show()
    denominator = df.shape[0] # how many observations am I working with
    rank: int
    for rank, count in df["rank_appeared"].value_counts().iteritems():
        rank_click_prob[rank] = 1 - (count/denominator)


    #  add these interaction counts to my default dict with this apply trick
    df[["reference","rank_appeared"]].apply(create_entry,axis=1)
    df["impressions"].apply(count_denom)

    # then do the division num/denom and output to a new dataframe
    safe_divide = lambda x,y: x / y if (y > 0) else 0
    record_format  = [{"item_id": id, "ctr":safe_divide(dict["num"],dict["denom"]), "ctr_prob": safe_divide(dict["num_pw"],dict["denom_pw"])} for id,dict in compute_ctr.items()]
    clicks = pd.DataFrame.from_records(record_format)

    # i want the clicks df to look like:
    # id | ctr | ctr_prob
    # AB | 0.1 | 0.03

    # send this to csv
    clicks.to_csv("data/trivago/item_clicks.csv",index=False)


if __name__ == "__main__":
    main()
