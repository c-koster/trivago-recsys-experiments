"""
I split up the load and train steps into different pieces -- this experiment file contains
some machinery to load in train/vali/test data, tune hyper-parameters for a model, and
evaluate a model (using MRR or AUC).

1. Load in the data and split it up

2. get scaled matrices out of the entire train vali and test sets. Remember not to fit_transform
vali and test.

3. try random forest and MLPClassifier models for learning click probability, picking the hyperparameters which
yield the best MRR on the validation set

4. evaluate the best ones on the test set
"""
# srun --pty /bin/bash

from typing import Dict, List, Set, Optional, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass # experiment result tuple


import pandas as pd
import numpy as np


# sklearn models and processing. It would be really cool if I didn't have to
# decompose my pandas df into a dictionary to then run dictvectorizer and standardscaler
# on it but I'm getting some weird error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# models --
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import ClassifierMixin

RANDOM_SEED = 42
n_rand = 3

data_all = pd.read_parquet("data/trivago/data_all.parquet", engine="pyarrow")
print("loaded data",flush=True)
# split the data back into three frames
train = data_all[data_all.grp == 0]
vali = data_all[data_all.grp == 1]
test = data_all[data_all.grp == 2]
print("split data",flush=True)
feature_names = set(data_all.columns) - set(["y", "q_id", "grp", "choice_idx", "is_advantaged_user"])


# helpers --
def safe_mean(input: List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)

def auc(m: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> float:
    """ I've typed out these two lines 10_000 times so now it's one line """
    m_pred = m.predict_proba(X)[:, 1].ravel()
    return roc_auc_score(y_true=y, y_score=m_pred)


fscale = StandardScaler()
X_train = fscale.fit_transform(train[feature_names])
y_train = train["y"].array
print("Fit the StandardScaler",flush=True)

f: ClassifierMixin = LogisticRegression()
f.fit(X_train, y_train)
train_auc = auc(f,X_train,y_train) # how well did I memorize the training data


X_vali = fscale.transform(vali[feature_names])
y_vali = vali["y"].array
vali_auc = auc(f,X_vali,y_vali)

X_test = fscale.transform(test[feature_names])
y_test = test["y"].array


# how well did my model learn the data
print("\n---Data Shape---")
print("train shape: {}\n vali shape {}\n test shape {}\n".format(str(X_train.shape),str(X_vali.shape),str(X_test.shape)))
print("train AUC: {:3f}\n vali AUC {:3f}".format(train_auc,vali_auc),flush=True)

@dataclass
class ExperimentResult: # fancy tuple with its own print function
    model: ClassifierMixin
    params: Dict[str,str]
    # metrics
    train_auc: float
    vali_auc: float
    mrr_train: float
    mrr_vali: float

    def outputs(self) -> None:
        print("Model params:",self.params)
        print("Results\n-------\n train_auc: {:3f}\nmrr_train: {:3f}\n vali_auc: {:3f}\n vali_mrr: {:3f}".format(self.train_auc,self.mrr_train,self.vali_auc,self.mrr_vali))
        if hasattr(self.model, 'feature_importances_'):
            print(
                "Feature Importances:",
                sorted(
                    zip(feature_names, [int(1000*i) for i in self.model.feature_importances_]), key=lambda tup: tup[1], reverse=True,
                ),
            )
        else:
            print("Not explainable.")


def tune_RF_model() -> ExperimentResult:
    experiments: List[ExperimentResult] = []
    for rnd in tqdm(range(n_rand)): # random seed loop
        for crit in ["gini", "entropy"]:
            for d in [8,16,32, None]:
                params: Dict[str,str] = {
                    "random_state": RANDOM_SEED + rnd,
                    "criterion": crit,
                    "max_depth": d,
                }
                m = RandomForestClassifier(**params)
                m.fit(X_train, y_train)

                train_auc = auc(m,X_train,y_train)
                vali_auc = auc(m,X_vali,y_vali)

                train_mrr = safe_mean(compute_clickout_RR(m,train))
                vali_mrr = safe_mean(compute_clickout_RR(m,vali))

                result = ExperimentResult(m,params,train_auc,vali_auc,train_mrr,vali_mrr)
                experiments.append(result)


    return max(experiments, key = lambda tup: tup.mrr_vali)

def tune_MLP_model() -> ExperimentResult:
    experiments: List[ExperimentResult] = []
    for rnd in tqdm(range(n_rand)): # random seed loop
        for layer in [(32,), (16,16,), (16,16,16,)]:
            for activation in ['logistic','relu']:
                params: Dict[str,str] = {
                    "hidden_layer_sizes": layer,
                    "random_state": RANDOM_SEED + rnd,
                    "activation":activation,
                    "solver": "lbfgs",
                    "max_iter": 1e4,
                    "alpha": 0.0001,
                }
                m = MLPClassifier(**params)

                m.fit(X_train, y_train)

                train_auc = auc(m,X_train,y_train)
                vali_auc = auc(m,X_vali,y_vali)

                train_mrr = safe_mean(compute_clickout_RR(m,train))
                vali_mrr = safe_mean(compute_clickout_RR(m,vali))

                result = ExperimentResult(m,params,train_auc,vali_auc,train_mrr,vali_mrr)
                experiments.append(result)

    return max(experiments, key = lambda tup: tup.mrr_vali)


# ok now evaluate the model on the metric that we care about: Mean Reciprocal Rank (MRR)
def calc_RR(sorted_item_list: List[bool]) -> float:
    """
    The way I've set this up: we want the True label (the item that the user really clicked)
    to appear at the first rank. Return the rank where it really appears
    """
    return 1 / (sorted_item_list.index(True) + 1)



def compute_clickout_RR(model: ClassifierMixin, data: pd.DataFrame) -> List[float]:
    """
    Mean Reciprocal Rank:
    """
    #unique_query_ids: List[str] = list(data.q_id.unique())
    reciprocal_ranks: List[float] = []

    grouped = data.groupby("q_id")
    for query_str, query in grouped: # loop over each query that i just grouped. Each iteration is a (q_id, DataFrame) pair

        y_qid = np.array(query["y"]).ravel()
        X_qid = fscale.transform(query[feature_names])

        if True not in y_qid: # if this example is unlabeled -- then we should skip it.
            continue

        # this outputs two values -- we want the probability of the second class (found at each row of  index 1 in this matrix)
        qid_scores = model.predict_proba(X_qid)[:,1].ravel()

        # predict on features X and re-sort the items.. return a sorted list of item ids
        impressions_shuffled = [i[0] for i in sorted(zip(y_qid, qid_scores), key=lambda tup: tup[1],reverse=True)]
        rr = calc_RR(impressions_shuffled) # Score based upon the index where you found the relevant item.
        reciprocal_ranks.append(rr)
        # we don't quite care about the real choice in creating features here
        # because we're not evaluating click accuracy, we're using a listwise metric

    return reciprocal_ranks


MRR_train = safe_mean(compute_clickout_RR(f,train))
MRR_vali = safe_mean(compute_clickout_RR(f,vali))
print("MRR_train: {:3f}\nMRR_vali: {:3f}\n".format(MRR_train,MRR_vali))

test_adv = test[test.is_advantaged_user == 1]
test_disadv = test[test.is_advantaged_user == 0]

# get the reciprocal ranks for all clickouts in each dataset
test_adv_ranks = compute_clickout_RR(f,test_adv)
test_disadv_ranks = compute_clickout_RR(f,test_disadv)

MRR_test_adv = safe_mean(test_adv_ranks)
MRR_test_disadv = safe_mean(test_disadv_ranks)
MRR_test_all = safe_mean(test_adv_ranks + test_disadv_ranks)
print("LR results on test set, split by n_sessions in a given user's profile")
print("MRR_advantaged: {:3f}\nMRR_disadvantaged: {:3f}\nMRR_all: {:3f}\n".format(MRR_test_adv, MRR_test_disadv, MRR_test_all))


print("trying a bunch of RF models")
rf = tune_RF_model()
rf.outputs()

print("and some MLP classifiers")
nn = tune_MLP_model()
nn.outputs()


# random forest listwise stats over the test set
test_adv_ranks_rf = compute_clickout_RR(rf.model,test_adv)
test_disadv_ranks_rf = compute_clickout_RR(rf.model,test_disadv)

# do this so I don't evaluate the whole test set separately from the subdivided
# test  set -- MRR takes a while to compute
MRR_test_adv_rf = safe_mean(test_adv_ranks_rf)
MRR_test_disadv_rf = safe_mean(test_disadv_ranks_rf)
MRR_test_all_rf = safe_mean(test_adv_ranks_rf + test_disadv_ranks_rf)

rf_results = {"all": MRR_test_all_rf, "adv": MRR_test_adv_rf, "dis": MRR_test_disadv_rf}
print("\nRF TEST-SET RESULTS: \ntotal MRR: {all:3f}\nadvangaged MRR: {adv:3f}\ndisadvangaged MRR: {dis:3f}".format(**rf_results))

# same but for the MLP/nn
test_adv_ranks_nn = compute_clickout_RR(nn.model,test_adv)
test_disadv_ranks_nn = compute_clickout_RR(nn.model,test_disadv)

MRR_test_adv_nn = safe_mean(test_adv_ranks_nn)
MRR_test_disadv_nn = safe_mean(test_disadv_ranks_nn)
MRR_test_all_nn = safe_mean(test_adv_ranks_nn + test_disadv_ranks_nn)

nn_results = {"all": MRR_test_all_nn, "adv": MRR_test_adv_nn, "dis": MRR_test_disadv_nn}
print("\nNN TEST-SET RESULTS: \ntotal MRR: {all:3f}\nadvangaged MRR: {adv:3f}\ndisadvangaged MRR: {dis:3f}".format(**nn_results))
