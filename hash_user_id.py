"""
Second experiment:

We think our user-aware recommender system is pretty powerful. But what happens when we pretend
that all users are unique, and rely on session context/item context alone for making recommendations?

This will be annswered in the second experiment file -- but first we have to make identical
train/test.csv files with re-coded user_ids to pretend that each session comes from a 'new user'.

I do this by setting the user id to the hash of the session_id.
"""
from hashlib import md5
import pandas as pd

def hash_session(id: str) -> str:
    """ Input a session id and perform md5, then return a string of the hex digest. """
    bytes = id.encode('utf-8')
    return md5(bytes).hexdigest()


def convert_file(what: str) -> None:
    assert what in ["train","test"]
    df = pd.read_csv("data/trivago/{}.csv".format(what))
    df["user_id"] = df["session_id"].apply(hash_session)
    df.to_csv("data/trivago/{}_hashed.csv".format(what),index=False)

convert_file("train")
convert_file("test")
