"""
This is an *almost* replica of the load_trivago file. I'll think about making a functions
file and import the common pieces from there if I end up passing this code to someone
"""
RANDOM_SEED = 42

#basic
import csv
from os import path
import random
import math
from time import time

# matrix
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import components

import matplotlib.pyplot as plt


# fancy python styling and error bars
from typing import Dict, List, Set, Optional, Any, Tuple
from tqdm import tqdm

from dataclasses import dataclass


# helpers --
def safe_mean(input: List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)

def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))

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

def hotel_sim_embed(id_x: str, id_y: str) -> float:
    """
    Look up the ids of the hotels from the dict and return their embeddding similarity
    """
    try:
        sim: float = w2v_model.wv.similarity(id_x, id_y)
        return sim
    except KeyError:
        return 0.0


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


@dataclass(unsafe_hash=True) #TypeError: unhashable type: 'Session' just don't use me as a key
class Session:
    start: int
    user_id: str
    session_id: str
    interactions: List[Interaction]
    is_advantaged_user: bool


    def set_user_id(self,num: int) -> "Session":
        """
        This helps us with the *extra training data* strategy where we want copies of
        our session data which don't align with user profiles -- to prevent over-learning
        user-based features while training.
        """
        new_user_id: str = self.user_id + "abc" + str(num)
        new_session_id: str = self.session_id + "abc" + str(num)
        new_session: Session = Session(self.start,new_user_id,new_session_id,self.interactions)
        return new_session

    def append_interaction(self, o: Interaction) -> None:
        """ helper """
        self.interactions.append(o)


# makes less sense to make a dataclass here because user profiles change often
"""
First rework goes here -- I no longer have user profiles so this "user profile" object can be removed completely.
Using session.interactions[:step] has the same effect of updating features continuously in real time.
"""

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
        if is_clickout:
            i = Interaction(t,d["action_type"], str(d["reference"]),
            is_clickout, d["impressions"].split("|"), [float(i) for i in d["prices"].split("|")])
        else:
            i = Interaction(t,d["action_type"],str(d["reference"]),is_clickout,[],[])

        interaction_list.append(i)

    # a person is an advantaged user if they had more than 2 sessions in the training set
    return Session(interaction_list[0].timestamp,df[0]["user_id"],df[0]["session_id"],interaction_list,(df[0]["nsessions"] > 2))


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
    hotel_sims_embed = [hotel_sim_embed(current_choice_id, o.action_on) for o in prev_clickouts] if item_exists else [0]


    dst_shortest_path: float
    try:
        if nx.has_path(G,current_choice_id, session.user_id):
            dst_shortest_path = sigmoid(nx.shortest_path_length(G,current_choice_id, session.user_id, weight = None))
        else:
            dst_shortest_path = sigmoid(100.0)
    except nx.NodeNotFound:
        dst_shortest_path = -1.0

    features: Dict[str,Any] = { #type:ignore
        # session-based features
        "time_since_start": current_timestamp - session.start,
        "time_since_last_clickout": current_timestamp - last_clickout.timestamp if last_clickout else 0,
        "diff_price_mean": current_price - safe_mean(session.interactions[step].prices),
        "last_price_diff": current_price - last_clickout.prices[last_clickout.get_clicked_idx()] if last_clickout else 0,
        "reciprocal_choice_rank": 1 / (choice_idx + 1), # rank starts at 1 index starts at
        # person clicks 1 then 4... maybe they will try 7 next
        "predicted_next_click": 2 * last_clickout.get_clicked_idx() - prev_clickouts[-2].get_clicked_idx() + 1 if len(prev_clickouts) > 1 else last_clickout.get_clicked_idx() + 2 if last_clickout else 1,
        # position to previous clickout position
        "delta_position_last_position": (choice_idx) - last_clickout.get_clicked_idx() if last_clickout else 0,
        # z-score (?) difference between price and average price of clicked hotels
        "avg_price_sim": current_price - safe_mean([o.prices[o.get_clicked_idx()] for o in prev_clickouts]) if last_clickout else 0,
        "prev_impression_sim": jaccard(set(last_clickout.impressions),set(current_clickout.impressions)) if last_clickout else 0,
        # user-item or item-item features --
        # roll through the previous items interacted in the session and average their jaccard similarity to the current item
        "item_item_sim": safe_mean(hotel_sims),
        "item_item_embed_sim": safe_mean(hotel_sims_embed),
        "item_ctr": id_to_hotel[current_choice_id].ctr if item_exists else 0,
        "item_ctr_prob": id_to_hotel[current_choice_id].ctr_prob if item_exists else 0,
        # user-based features (these build on previous sessions or in conjunction with current sessions) --
        "path_user_item": dst_shortest_path
    }
    return features


def build_graph(session_ids: List[str]) -> nx.Graph:
    """
    Build a graph for collaborative filtering using a list of session ids.
    """
    G = nx.Graph()

    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]
    for s in sessions:
        clickouts = [o for o in s.interactions if o.is_clickout]
        G.add_edges_from([(o.action_on, s.user_id) for o in clickouts])

    return G


from gensim.models import Word2Vec
import multiprocessing

def make_w2v(session_ids: List[str]) -> Word2Vec:
    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]
    sequences: List[str] = []
    for s in sessions:
        clickouts = [o for o in s.interactions if o.is_clickout]
        sequences += [o.impressions for o in clickouts]

    n_cores = multiprocessing.cpu_count()

    # 1. initialise model with params
    w2v_model = Word2Vec(min_count=1,
                     window=4,
                     vector_size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=8,
                     workers=n_cores-1)

    # 2. create the vocabulary
    w2v_model.build_vocab(sequences, progress_per=10000)

    t = time()
    w2v_model.train(sequences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    return w2v_model


def collect(what: str, session_ids: List[str]) -> SessionData:
    """
    This function takes as input a filename and will return a SessionData object containing
    a list of clickout features, a list of labels, and methods to convert each into matrices.
    """
    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]

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

                        # feature extraction needs session, interaction info, and the index we were wondering about.
                        # we also pass whether we want to ignore user features. (defaults to false)
                        features = extract_features(s,step,index)
                        features["is_advantaged_user"] = 1 if s.is_advantaged_user else 0
                        #1 if len(users[s.user_id].sessions) > 2 else 0
                        q_id = "{}/{}".format(s.session_id, step)
                        qids.append(q_id)
                        examples.append({"q_id": q_id, "choice_idx":index, **features, "y":label})

                    G.add_edge(o.action_on, s.user_id) # then add the edge to the graph.. aka build it in real time.


    feature_names: List[str] = [i for i in features.keys()]
    return SessionData(pd.DataFrame.from_records(examples), qids, feature_names)


def load_session_dict(what: str) -> Dict[str,Session]:
    """
    This function be called on both train and test sets, so that I can get both a list of unique session ids to collect later,
    and a way to access these sessions (in collect) in  O(1) time.
    """

    sessions: List[Session] = []
    # nrows=1_000 for my laptop's sake
    df_interactions = pd.read_csv("data/trivago/{}_hashed.csv".format(what)) #type:ignore
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
# create an empty graph - nodes and edges will be built in real-time because we want to prevent crazy amounts of overfitting
G = nx.Graph()


# load in my item features --

print("Loading up item features")
if not path.exists("data/trivago/item_metadata_dense.csv"):
    import extract_hotel_features
    extract_hotel_features.main()


hotel_features_df = pd.read_csv("data/trivago/item_metadata_dense.csv") #type:ignore
hotel_features_df["item_id"] = hotel_features_df["item_id"].apply(str) # make sure the id is of type str, not int
hotel_features_df["properties"] = hotel_features_df["properties"].str.split("|").map(set)

d: Dict = hotel_features_df.to_dict("records")
for h in d: # loop over the dictionary version of this df
    id_to_hotel[h["item_id"]] = Hotel(**h)

"""
Code starts here to generate features, splits, and quickly-loadable matrix files. There
are two tasks:
    1. make a parquet file for user-aware train/eval later
    2. make another file to be used for a user-blind recommender. This is a little tricky because I still
    need to know if a user belongs to a 'advantaged' or 'disadvantaged' group. Pass in a parameter to
    feature extraction to ignore the user ?
"""
# part 1:
print("Loading Session Data")
sessions_train = load_session_dict("train")
sessions_vali = load_session_dict("validation")
sessions_test = load_session_dict("confirmation")

session_ids_train = list(sessions_train.keys())
session_ids_vali = list(sessions_vali.keys())
session_ids_test = list(sessions_test.keys())

sids_to_data = {**sessions_train, **sessions_vali,**sessions_test}

# get w2v
if not path.exists("data/trivago/query_sim_w2v.model"):
    make_w2v(session_ids_train).save("data/trivago/query_sim_w2v.model")
w2v_model = Word2Vec.load("data/trivago/query_sim_w2v.model")


train = collect("train",session_ids_train)
session_ids_vali_test = session_ids_test + session_ids_vali
session_ids_vali_test.sort(key=lambda x: sids_to_data[x].start)
valitest = collect("vali/test", session_ids_vali_test)
#test = collect("test", session_ids_test)

train.data["grp"] = 0
vali_or_test = lambda x: 2 if x.split("/")[0] in session_ids_test else 1
valitest.data["grp"] = valitest.data["q_id"].apply(vali_or_test)
#test.data["grp"]  = 2


frames: List[pd.DataFrame] = [train.data, valitest.data]
df_out = pd.concat(frames)
print("Writing a df with pyarrow. It has dimensions {}. ".format(df_out.shape))

# dump dataset and put my experiments in a different file
df_out.to_parquet("data/trivago/data_all_user_blind.parquet", engine="pyarrow")

# quick sanity check: is the number of interactions over ALL sessions equal to
# the number of interactions in all user profiles by the end of feature generation?


# I should build blind df differently  ... directly from datafile with a 'is_advantaged_user label'
# do we want a new train/vali split?
