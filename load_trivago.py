"""
I have a Trivago dataset with user sessions. What task am I trying to solve?

I want to leverage user profile and session data to create a model which estimates the
probability that a person will click-out on a hotel, and use this probability to sort
search results.

What do my labels look like?
    - my positive labels (1) are click-out events and negative labels are ones which
    a user did not click (0). So, because there are 25 resuls and one click-out event,
    there will be 1 positive example for 24 negatives.

What information do I have while crafting features?
    - all of the information which precedes the clickout action in the session.
    - entire user profile. see 2019 recsys winners for item similarity with items
    which a user has previously looked at.


Extract labels and features as follows:
    - for each session id: create an ordered list of sessions.
    - for each user id: construct a profile of all sessions done by that user
    - roll through each session

"""

#basic
import csv
from datetime import datetime

# matrix
import pandas as pd
import numpy as np

# fancy python styling and error bars
from typing import Dict, List, Set, Optional, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass


# First define classes to make handling data a little easier.
@dataclass
class Hotel:
    props: Set[str]

@dataclass
class Interaction:
    timestamp: datetime
    action_type: str
    action_on: str
    is_clickout: bool
    # if it's a clickout event we have more data.
    # what's the best way to say this with dataclasses?
    impressions: List[str]
    prices: List[float]

@dataclass(unsafe_hash=True) #TypeError: unhashable type: 'Session' -- should be fine as we don't change this ever
class Session:
    start_timestamp: datetime
    user_id: str
    session_id: str
    interactions: List[Interaction]

@dataclass
class UserProfile:
    user_id: str
    sessions: List[Session]


# this one is a wrapper for training and test data types
@dataclass
class SessionData:
    Xs: List[Dict[str,str]]
    ys: List[bool]


# and functions

def create_session(df: List[Dict[str,str]]) -> Session:
    """
    This function takes a list of interactions and saves it as a session in object format.
    """
    interaction_list: List[Interaction] = []

    for d in df:
        i: Interaction
        is_clickout = ("clickout item" == d["action_type"])

        if is_clickout:
            i = Interaction(datetime.fromtimestamp(int(d["timestamp"])),d["action_type"],d["reference"],
            is_clickout, d["impressions"].split("|"), [float(i) for i in d["prices"].split("|")])
        else:
            i = Interaction(datetime.fromtimestamp(int(d["timestamp"])),d["action_type"],d["reference"],is_clickout,[],[])

        interaction_list.append(i)
    return Session(interaction_list[0].timestamp,df[0]["user_id"],df[0]["session_id"],interaction_list)

def extract_features(x: int) -> None:
    """

    """
    pass

def collect(what: str) -> SessionData:
    """
    This function will return a SessionData object which contains a feature and labels list,
    and methods to turn both into matrices.
    """
    assert(what in ["train","test"])

    sessions: List[Session] = []

    df_interactions = pd.read_csv("data/trivago/{}.csv".format(what),nrows=10_000) #type:ignore
    # appply the "save_session" function to each grouped item/session
    # but first turn each group from a df into a list of dictionaries
    A = lambda x: sessions.append(create_session(x.to_dict("records"))) #type:ignore
    df_interactions.groupby(by="session_id").apply(A)

    if what == "train":
        print("Building user profiles") # how do I want to make user profiles?
        for s in sessions: # loop through sessions
            # try to add each session to session.user_id

            uid = s.user_id # (use this many times)
            try:
                users[uid].sessions.append(s) # try to add the session
            except KeyError:
                # but if it doesn't work, create a new user at that address.
                users[uid] = UserProfile(uid,[s])

    print("Rolling through sessions/creating labeled feature vectors for {}.csv".format(what))

    Xs: List[Dict[str,str]] = []
    ys: List[bool] = []

    # TODO
    # for each session
        # for each interaction in the session
            # if it's of type "clickout", e.g. o.is_clickout
                # create a positive training example and k negative samples
                # extract features for each and add to x

    assert(len(Xs) == len(ys))
    return SessionData(Xs,ys)

# globals
item_properties_all: Dict[str,Hotel] = {}
users: Dict[str,UserProfile] = {} # this map ids to UserProfile objects (which are just sets of sessions)

# load in my item features
with open("data/trivago/item_metadata.csv") as file:
    reader = csv.DictReader(file)
    dict: Dict[str,str]
    for dict in tqdm(reader,total=927143):
        id = dict["item_id"]
        props: List[str] = dict["properties"].split("|")
        item_properties_all[id] = Hotel(set(props))

traindata = collect("train")
testdata = collect("test")
