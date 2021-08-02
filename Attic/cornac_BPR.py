"""
Collaborative filtering: Use user-item interactions to help in recommendations.

Code adapted from a https://github.com/microsoft/recommenders notebook on the movielens dataset.

a few things to note. That the examples I’m seeing in the notebook are user/movie
interactions in which each user has *only one* interaction with a movie, and this interaction
has a rating between 1 and 5. The dataset I’m working with has potentially multiple interactions,
and these are not rated.

My first task is to code values for click-outs and item-information clicks, and sum these so that only
one interaction exists for any user and hotel pair (I need a bipartite graph, not a bipartite multigraph).
"""

import sys
import os
import cornac

import pandas as pd

from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
from reco_utils.common.timer import Timer
from reco_utils.common.constants import SEED

# top k items to recommend
TOP_K = 25

# Model parameters
NUM_FACTORS = 200
NUM_EPOCHS = 100

types = []
train = pd.read_csv("data/trivago/train.csv") #type:ignore

check_valid = lambda x: x.isnumeric()
train["is_item_interaction"] = train["reference"].apply(check_valid)
train = train[train.is_item_interaction == True]


rating_func = lambda x: 1 if x == 'clickout item' else 0.2
train["rating"] = train["action_type"].apply(rating_func)
train["group_id"] = train["user_id"] + "/" + train["reference"]
train = train.groupby(["group_id"]).agg({'user_id': 'first', 'reference': 'first', 'rating':'sum'})
print(train.columns)


# get my pretend movielens dataset into cornac format -- and p
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)

print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))


bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)

with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))


with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='user_id', itemcol='reference', remove_seen=True)
print("Took {} seconds for prediction.".format(t))

print(all_predictions.head())
