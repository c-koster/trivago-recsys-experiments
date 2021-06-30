"""
Compute click-through ratios for hotels
"""
from typing import Dict, List

import pandas as pd
from collections import defaultdict


compute_ctr: Dict = defaultdict(lambda: {"num":0, "denom":0})

# helpers for main function -- these get applied to df rows
def create_entry(item_id:str) -> None:
    e = compute_ctr[item_id]
    e["num"] += 1


def count_denom(L: List) -> None:
    for i in L:
        compute_ctr[i]["denom"] +=1


def main() -> None:
    df = pd.read_csv("data/trivago/train.csv")
    df = df[df.action_type == "clickout item"]
    df = df[df.reference != "nan"]


    df["impressions"] = df["impressions"].str.split("|").map(list)

    # put add to my default dict with this apply trick
    df["reference"].apply(create_entry)
    df["impressions"].apply(count_denom)

    safe_divide = lambda x,y: x / y if (y> 0) else 0
    clicks = pd.DataFrame.from_records([{"item_id": id, "ctr":safe_divide(dict["num"],dict["denom"])} for id,dict in compute_ctr.items()])

    # i want the df to look like
    # id | ctr
    # AB | 0.1
    clicks.to_csv("data/trivago/item_clicks.csv",index=False)



if __name__ == "__main__":
    main()
