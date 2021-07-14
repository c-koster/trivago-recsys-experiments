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
global err_count
err_count = 0
RANDOM_SEED = 42

#basic
import csv
from os import path
import random
#from datetime import datetime

# matrix
import pandas as pd
import numpy as np

# fancy python styling and error bars
from typing import Dict, List, Set, Optional, Any, Tuple
from tqdm import tqdm

from dataclasses import dataclass

from sklearn.model_selection import train_test_split


# helpers --
def safe_mean(input: List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)

def jaccard(lhs: Set[str], rhs: Set[str]) -> float:
    isect_size = sum(1 for x in lhs if x in rhs)
    union_size = len(lhs.union(rhs))
    return isect_size / union_size

def hotel_sim(id_x: str, id_y: str) -> float:
    """
    Look up the ids of the hotels from the dict and return the jaccard similarity of their properties
    """
    try:
        x_props = id_to_hotel[id_x].properties
        y_props = id_to_hotel[id_y].properties
        return jaccard(x_props,y_props)
    except KeyError:
        # fix these weird dictionary misses
        # print(id_x,id_y)
        return 0


# First define classes to make handling data a little easier.
@dataclass
class Hotel:
    item_id: str
    properties: Set[str]
    cat: str
    rating: float
    stars: float
    ctr: float
    ctr_prob: float




@dataclass
class Interaction:
    timestamp: int
    action_type: str
    action_on: str
    is_clickout: bool
    # if it's a clickout event we have more data.
    # what's the best way to say this with dataclasses?
    impressions: List[str]
    prices: List[float]

    def get_clicked_idx(self) -> int:
        """
        Helper to get the index of the clicked item, which is helpful for feature extraction.
        Be careful with this method for 2 reasons:
        1. if it's called on an interaction instance which is not a clickout, raise an error.
        2. using this method improperly in the feature extractor can generate the label for the current prediction task.
            for example: adding this line gives us an auc of 1.
            cheat = (choice_idx == current_clickout.get_clicked_idx())
        """
        assert(self.is_clickout)
        idx: int
        try:
            idx = self.impressions.index(self.action_on)
        except ValueError:
            idx = -1
        return idx

@dataclass(unsafe_hash=True) #TypeError: unhashable type: 'Session' -- should be fine as we don't change this ever
class Session:
    start: int
    user_id: str
    session_id: str
    interactions: List[Interaction]

    def set_user_id(self,num: int) -> "Session":
        """
        Better to
        """
        new_user_id: str = self.user_id + "abc" + str(num)
        new_session_id: str = self.session_id + "abc" + str(num)
        new_session: Session = Session(self.start,new_user_id,new_session_id,self.interactions)
        return new_session


@dataclass
class UserProfile:
    user_id: str
    sessions: List[Session]
    unique_interactions: Set[str]


# this one is a wrapper for training and test data types
@dataclass
class SessionData:
    data: pd.DataFrame
    qids: List[str]
    feature_names: List[str]

    def get_session(self, session_id: str) -> Session:
        """
        Helps me commpute mean reciprocal rank later on.
        """
        # access the session dict, then skip to the corresponding step in that session's interaction list
        s_temp: Session = sids_to_data[session_id]
        return s_temp


def create_session(df: List[Dict[str,str]]) -> Session:
    """
    This function takes a list of interactions and saves it as a session in object format.
    """
    interaction_list: List[Interaction] = []

    for d in df:
        i: Interaction
        is_clickout = ("clickout item" == d["action_type"])
        t = int(d["timestamp"])#datetime.fromtimestamp(int(d["timestamp"]))
        global err_count
        if is_clickout:
            if type(d["reference"]) == type(1.0): #  meaning that there is a clickout but it is a NaN. don't know whow to interpret this TODO
                err_count +=1
            i = Interaction(t,d["action_type"], str(d["reference"]),
            is_clickout, d["impressions"].split("|"), [float(i) for i in d["prices"].split("|")])
        else:
            i = Interaction(t,d["action_type"],str(d["reference"]),is_clickout,[],[])

        interaction_list.append(i)
    return Session(interaction_list[0].timestamp,df[0]["user_id"],df[0]["session_id"],interaction_list)


def extract_features(session: Session, step: int, choice_idx: int) -> Dict[str,Any]:
    """
    Feature extraction for one session step of action type 'clicked out'
    """
    current_clickout = session.interactions[step]
    current_timestamp = current_clickout.timestamp # shorthand I'll use this a lot lot
    current_price = current_clickout.prices[choice_idx]
    current_choice_id = current_clickout.impressions[choice_idx]

    prev_clickouts: List[Interaction] = [o for o in session.interactions[:step] if o.is_clickout]
    # last_clickout is really useful for extracting features
    # -- this will be set to None if there was no clickout
    last_clickout = prev_clickouts[-1] if len(prev_clickouts) else None

    item_exists: bool
    try:
        id_to_hotel[current_choice_id]
        item_exists = True
    except KeyError:
        item_exists = False
        #print("Key Error... {} does not exist in our dict".format(current_choice_id))


    hotel_sims = [hotel_sim(current_choice_id, o.action_on) for o in session.interactions[:step] if o.action_on.isnumeric()]

    user_hotel_unique_sims = [hotel_sim(current_choice_id, hotel_id) for hotel_id in users[session.user_id].unique_interactions] if session.user_id in users.keys() else [0]
    unique_item_sim = safe_mean(user_hotel_unique_sims)
    features: Dict[str,Any] = { #type:ignore

        # these are cheating, remove eventually --
        "diff_now_end": len(session.interactions) - step,
        # session-based features
        "time_since_start": current_timestamp - session.start,
        "time_since_last_clickout": current_timestamp - last_clickout.timestamp if last_clickout else 0,
        "diff_price_mean": current_price - safe_mean(session.interactions[step].prices),
        "last_price_diff": current_price - last_clickout.prices[last_clickout.get_clicked_idx()] if last_clickout else 0,
        "reciprocal_choice_rank": 1 / (choice_idx + 1), # rank starts at 1 index starts at
        # position to previous clickout position
        "delta_position_last_position": (choice_idx) - last_clickout.get_clicked_idx() if last_clickout else 0,
        # z-score (?) difference between price and average price of clicked hotels
        "avg_price_sim": current_price - safe_mean([o.prices[o.get_clicked_idx()] for o in prev_clickouts]) if last_clickout else 0,
        "prev_impression_sim": jaccard(set(last_clickout.impressions),set(current_clickout.impressions)) if last_clickout else 0,
        # user-item or item-item features --
        # roll through the previous items interacted in the session and average their jaccard similarity to the current item
        "item_item_sim": safe_mean(hotel_sims),
        "item_ctr": id_to_hotel[current_choice_id].ctr if item_exists else 0,
        "item_ctr_prob": id_to_hotel[current_choice_id].ctr_prob if item_exists else 0,
        # user-based features (these build on previous sessions or in conjunction with current sessions) --
        "unique_item_interact_by_user": unique_item_sim
        #"time_since_last_interact_this_item":1
    }
    return features



def collect(what: str, session_ids: List[str], create_examples: float = 0.0) -> SessionData:
    """
    This function takes as input a filename and will return a SessionData object containing
    a list of clickout features, a list of labels, and methods to convert each into matrices.
    """
    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]

    if create_examples > 0: # TODO create examples is presently broken

        # add another session where we scramble the user ID so the model doesn't overfit
        sessions_duplicates, _ = train_test_split(sessions,train_size=0.1,random_state=RANDOM_SEED)
        [sessions.append(s_dup.set_user_id(num)) for num, s_dup in enumerate(sessions_duplicates)]
        print("added {} duplicates to our training data to prevent overfitting".format(len(sessions_duplicates)))

    print("Rolling through ({}) sessions/creating labeled feature vectors for {} sessions".format(len(sessions), what))

    examples: List[Dict[str,str]] = []
    qids: List[str] = []


    features: Dict[str,Any]

    for s in tqdm(sessions): # for each session -- --
        for step, o in enumerate(s.interactions): # for each interaction in the session


            # if it's of type "clickout", e.g. o.is_clickout
            if o.is_clickout:
                if o.get_clicked_idx() != -1:
                    # create a positive training example and 24 negatives... also extract features for each and add to x
                    for index, choice in enumerate(o.impressions):
                        label = (choice == o.action_on)

                        features = extract_features(s,step,index) # feature extraction needs session, interaction info, and
                        q_id = "{}/{}".format(s.session_id, step)
                        qids.append(q_id)
                        examples.append({"q_id": q_id, "choice_idx":index, **features, "y":label})


    feature_names: List[str] = [i for i in features.keys()]
    return SessionData(pd.DataFrame.from_records(examples), qids, feature_names)


def load_session_dict(what: str) -> Dict[str,Session]:
    """
    This function be called on both train and test sets, so that I can get both a list of unique session ids to collect later,
    and a way to access these sessions (in collect) in  O(1) time.
    """
    assert(what in ["train","test"])

    sessions: List[Session] = []
    # nrows=1_000 for my laptop's sake
    df_interactions = pd.read_csv("data/trivago/{}.csv".format(what)) #type:ignore
    # appply the "save_session" function to each grouped item/session
    # but first turn each group from a df into a list of dictionaries
    A = lambda x: sessions.append(create_session(x.to_dict("records"))) #type:ignore
    df_interactions.groupby(by="session_id").apply(A)

    sessions_dict: Dict[str,Session] = {}
    for s in sessions:
        sessions_dict[s.session_id] = s
    return sessions_dict

# globals
id_to_hotel: Dict[str,Hotel] = {}
users: Dict[str,UserProfile] = {} # this map ids to UserProfile objects (which are just sets of sessions)

# load in my item features --

if not path.exists("data/trivago/item_metadata_dense.csv"):
    import extract_hotel_features
    extract_hotel_features.main()

hotel_features_df = pd.read_csv("data/trivago/item_metadata_dense.csv") #type:ignore
hotel_features_df["item_id"] = hotel_features_df["item_id"].apply(str) # make sure the id is of type str, not int
hotel_features_df["properties"] = hotel_features_df["properties"].str.split("|").map(set)

d: Dict = hotel_features_df.to_dict("records")
for h in d: # loop over the dictionary version of this df
    id_to_hotel[h["item_id"]] = Hotel(**h)

sessions_tv = load_session_dict("train") # need to split ids by train and vali
sessions_test = load_session_dict("test")
session_ids_test = list(sessions_test.keys())

session_ids_train, session_ids_vali = train_test_split(list(sessions_tv.keys()),train_size=0.9,random_state=RANDOM_SEED)

print("Building user profiles") # how do I want to make user profiles?

s_id: str
for s_id in session_ids_train: # loop through sessions
    s: Session = sessions_tv[s_id]
    # try to add each session to session.user_id
    hotels_in_session: Set[str] = set([o.action_on for o in s.interactions if o.action_on != "nan"])

    uid = s.user_id # (use this many times)
    try:
        users[uid].sessions.append(s) # try to add the session
        users[uid].unique_interactions.update(hotels_in_session)
    except KeyError:
        # but if it doesn't work, create a new user at that address.
        users[uid] = UserProfile(uid,[s],hotels_in_session)

sids_to_data = {**sessions_tv,**sessions_test} # dangerous maybe to put train and test in the same dict?

train = collect("train",session_ids_train) # create_examples=0.1
vali = collect("vali",session_ids_vali)
test = collect("test",session_ids_test)


train.data["grp"] = 0
vali.data["grp"]  = 1
test.data["grp"]  = 2

frames: List[pd.DataFrame] = [train.data, vali.data, test.data]
df_out = pd.concat(frames)
print("Writing a df with pyarrow. It has dimensions {}. ".format(df_out.shape))

# dump dataset and put my experiments in a different file
df_out.to_parquet("data/trivago/data_all.parquet", engine="pyarrow")
print("Done!")
