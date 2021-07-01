"""
Extract rating and categoricial features from the hotels file, and then make a new file.
Adapted from https://github.com/logicai-io/recsys2019
"""

import pandas as pd
import pickle
from os import path

from typing import Dict, Any

RATING_MAP = {"Satisfactory Rating": 1, "Good Rating": 2, "Very Good Rating": 3, "Excellent Rating": 4}
STAR_MAP = {"1 Star": 1, "2 Star": 2, "3 Star": 3, "4 Star": 4, "5 Star": 5}
HOTEL_CAT = {
    "Hotel": "hotel",
    "Resort": "resort",
    "Hostal (ES)": "hostal",
    "Motel": "motel",
    "House / Apartment": "house",
}

def densify(d: Dict[str,Any], properties):
    """
    Apply the mapping d to each row of properties
    """
    values = [None] * properties.shape[0]
    for i, p in enumerate(properties):
        for k in d:
            if k in p:
                values[i] = d[k]
    return values

def main() -> None:
    if not path.exists("data/trivago/item_clicks.csv"):
        import compute_ctr
        compute_ctr.main()


    df = pd.read_csv("data/trivago/item_metadata.csv")
    df["properties_temp"] = df["properties"].str.split("|").map(set)
    df["rating"] = densify(RATING_MAP, df["properties_temp"])
    df["stars"] = densify(STAR_MAP, df["properties_temp"])
    df["cat"] = densify(HOTEL_CAT, df["properties_temp"])

    df = df.drop(columns=["properties_temp"])
    ctr = pd.read_csv('data/trivago/item_clicks.csv')
    merged = pd.merge(df,ctr,on="item_id",how="left")
    merged = merged.fillna(value=0)
    merged.to_csv("data/trivago/item_metadata_dense.csv", index=False)
    return

if __name__ == "__main__":
    main()
