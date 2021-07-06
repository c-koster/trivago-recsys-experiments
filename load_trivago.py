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

# sklearn. I want this ML experimentation in a different file really
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.base import ClassifierMixin

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
        idx: int = self.impressions.index(self.action_on)
        assert(idx != -1)

        return idx

@dataclass(unsafe_hash=True) #TypeError: unhashable type: 'Session' -- should be fine as we don't change this ever
class Session:
    start: int
    user_id: str
    session_id: str
    interactions: List[Interaction]

@dataclass
class UserProfile:
    user_id: str
    sessions: List[Session]
    unique_interactions: Set[str]


# this one is a wrapper for training and test data types
@dataclass
class SessionData:
    examples: List[Dict[str,str]]
    labels: List[bool]
    session_dict: Dict[str,Session]
    qids: List[str]

    def fit_vectorizer(self) -> DictVectorizer:
        numberer = DictVectorizer(sort=True, sparse=False)
        numberer.fit(self.examples)
        return numberer

    def get_matrix(self, numberer: DictVectorizer) -> np.ndarray:
        return numberer.transform(self.examples)


    def get_ys(self) -> np.ndarray:
        return np.array(self.labels)

    def get_session(self, session_id: str) -> Session:
        """
        Helps me commpute mean reciprocal rank later on
        """
        # access the session dict, then skip to the corresponding step in that session's interaction list
        s_temp: Session = self.session_dict[session_id]
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
            if type(d["reference"]) == type(1.0): # TODO: what are these examples ??
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
        print("Key Error... {} does not exist in our dict".format(current_choice_id))


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
        "item_ctr": id_to_hotel[current_choice_id].ctr if item_exists else 0.0,
        "item_ctr_prob": id_to_hotel[current_choice_id].ctr_prob if item_exists else 0.0,
        # user-based features (these build on previous sessions or in conjunction with current sessions) --
        "unique_item_interact_by_user": unique_item_sim
        #"time_since_last_interact_this_item":1
    }
    return features



def collect(what: str) -> SessionData:
    """
    This function takes as input a filename and will return a SessionData object containing
    a list of clickout features, a list of labels, and methods to convert each into matrices.
    """
    assert(what in ["train","test"])

    sessions: List[Session] = []

    df_interactions = pd.read_csv("data/trivago/{}.csv".format(what),nrows=1_000) #type:ignore
    # appply the "save_session" function to each grouped item/session
    # but first turn each group from a df into a list of dictionaries
    A = lambda x: sessions.append(create_session(x.to_dict("records"))) #type:ignore
    def append_session(x, add_ambiguous_users: bool = False) -> None:
        dict = x.to_dict("records")
        sess = create_session([])
        sessions.append(sess)
        #if add_ambiguous_users:
        #random.seed(RANDOM_SEED)
        #if random.random() > 0.1:
        # create a session with a random user id


    df_interactions.groupby(by="session_id").apply(A)

    # for every ten sessions: add another where we scramble


    if what == "train":
        print("Building user profiles") # how do I want to make user profiles?
        for s in sessions: # loop through sessions
            # try to add each session to session.user_id
            hotels_in_session: Set[str] = set([o.action_on for o in s.interactions if o.action_on != "nan"])

            uid = s.user_id # (use this many times)
            try:
                users[uid].sessions.append(s) # try to add the session
                users[uid].unique_interactions.update(hotels_in_session)
            except KeyError:
                # but if it doesn't work, create a new user at that address.
                users[uid] = UserProfile(uid,[s],hotels_in_session)


    print("Rolling through ({}) sessions/creating labeled feature vectors for {}.csv".format(len(sessions), what))

    Xs: List[Dict[str,str]] = []
    ys: List[bool] = []
    qids: List[str] = []
    session_dict: Dict[str,Session] = {}


    for s in sessions: # for each session -- --
        session_dict[s.session_id] = s # save it to a dict

        for step, o in enumerate(s.interactions): # for each interaction in the session

            # if it's of type "clickout", e.g. o.is_clickout
            if o.is_clickout:
                # create a positive training example and 24 negatives... also extract features for each and add to x
                for index, choice in enumerate(o.impressions):
                    label = (choice == o.action_on)

                    features = extract_features(s,step,index) # feature extraction needs session, interaction info, and

                    qids.append("{}/{}".format(s.session_id, step))
                    Xs.append(features)
                    ys.append(label)


    return SessionData(Xs,ys,session_dict, qids)


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



train = collect("train")
#print("odd examples count after train: {}".format(err_count))

test = collect("test")
#print("odd examples count after test: {}".format(err_count))





# dump dataset and put this in a different file

from sklearn.linear_model import LogisticRegression
numberer = train.fit_vectorizer()
fscale = StandardScaler()
X_train = fscale.fit_transform(train.get_matrix(numberer))
y_train = train.get_ys()

#f = RandomForestClassifier()
f = LogisticRegression()
f.fit(X_train, train.get_ys())


train_pred = f.predict_proba(X_train)[:, 1].ravel()
train_auc = roc_auc_score(y_true=y_train, y_score=train_pred)
# how well did I memorize the training data


X_test = fscale.fit_transform(test.get_matrix(numberer))
y_test = test.get_ys()


test_pred = f.predict_proba(X_test)[:, 1].ravel()
test_auc = roc_auc_score(y_true=y_test, y_score=test_pred)
# how well did my model learn the data


print("\n---Results---")
print("training data's shape: {}".format(str(X_train.shape)))
print("train AUC: {:3f}\n test AUC: {:3f}\n".format(train_auc,test_auc))


# explain my model. works for linear only

print("Best Model Weights:")
#weights = f.feature_importances_.ravel()
weights = f.coef_.ravel()

for name, weight in sorted(zip(numberer.feature_names_,weights), key=lambda tup: tup[1],reverse=True):
    print("{}\t{}".format(name,weight))




# ok now evaluate the model on the metric that we care about: Mean Reciprocal Rank (MRR)
def calc_RR(sorted_item_list: List[str], id: str) -> float:
    assert(id in sorted_item_list)
    return 1 / (sorted_item_list.index(id) + 1)


def compute_clickout_RR(model: ClassifierMixin, data: SessionData) -> List[float]:
    """
    Mean Reciprocal Rank:
    """
    reciprocal_ranks: List[float] = []

    for x_idx, query_str in enumerate(data.qids):  # 1. get all the clickout ids, which should be of the form session_id/step
        unpack = query_str.split("/")
        session_id = unpack[0]
        step = int(unpack[1])
        # get_true_y -- true_y is an item id or I could set it up as a rank
        queried_session = data.get_session(session_id) # queried session
        o = queried_session.interactions[step]


        true_y = o.action_on
        if true_y not in o.impressions: # sometimes action_on is missing from the test set
            continue


        Xs_query = [extract_features(queried_session, step, index) for index, _ in enumerate(o.impressions)]
        # extract features takes a session, a step, and a choice index.


        X_qid = fscale.transform(numberer.transform(Xs_query))
        # this outputs two values -- we want the
        qid_scores = model.predict_proba(X_qid)[:,1].ravel()


        # predict on features X and re-sort the items.. return a sorted list of item ids
        impressions_shuffled = [i[0] for i in sorted(zip(o.impressions, qid_scores), key=lambda tup: tup[1],reverse=True) ]

        rr = calc_RR(impressions_shuffled,true_y)
        reciprocal_ranks.append(rr)
        # we don't quite care about the real choice in creating features here
        # because we're not evaluating click accuracy, we're using a listwise metric


        # Score based upon the index you found the relevant item.
    return reciprocal_ranks

MRR_train = safe_mean(compute_clickout_RR(f,train))
#MRR_test = np.mean(compute_clickout_RR(f,test))
print("MRR_train: {}\nMRR_test: {}".format(MRR_train,0))
