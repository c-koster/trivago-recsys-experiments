"""
This will look suspiciously like train eval. Some DIFFERENCES include:
    1. remove 'is_avantaged_user' as a feature but still evaluate each group individually
    2.

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
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import ClassifierMixin

import joblib

RANDOM_SEED = 42
n_rand = 3

data_all = pd.read_parquet("data/trivago/data_all_user_blind.parquet", engine="pyarrow")
print("loaded data",flush=True)

feature_names = set(data_all.columns) - set(["y", "q_id", "grp", "choice_idx","is_advantaged_user"])

# split the data
train = data_all[data_all.grp == 0]
vali = data_all[data_all.grp == 1]
test = data_all[data_all.grp == 2]

print("split data",flush=True)


# helpers --
def safe_mean(input: List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)

def auc(m: Any, X: np.ndarray, y: np.ndarray) -> float:
    """ I've typed out these two lines 10_000 times so now it's one line """
    m_pred = m.predict_proba(X)[:, 1].ravel()
    return roc_auc_score(y_true=y, y_score=m_pred)


fscale = StandardScaler()
X_train = fscale.fit_transform(train[feature_names])
y_train = train["y"].to_numpy()
print("Fit the StandardScaler",flush=True)

f: ClassifierMixin = LogisticRegression()
f.fit(X_train, y_train)
train_auc = auc(f,X_train,y_train) # how well did I memorize the training data


X_vali = fscale.transform(vali[feature_names])
y_vali = vali["y"].to_numpy()
vali_auc = auc(f,X_vali,y_vali)

X_test = fscale.transform(test[feature_names])
y_test = test["y"].to_numpy()


# how well did my model learn the data
print("\n---Data Shape---")
print("train shape: {}\nvali shape: {}\ntest shape: {}\n".format(str(X_train.shape),str(X_vali.shape),str(X_test.shape)))
print("train AUC: {:3f}\nvali AUC {:3f}".format(train_auc,vali_auc),flush=True)

@dataclass
class ExperimentResult: # fancy tuple with its own print function
    model: Any
    params: Dict[str,str]
    # metrics
    train_mrr: float
    vali_mrr: float

    def outputs(self) -> None:
        print("Model params:",self.params)
        print("Results\n-------\n\ntrain_mrr: {:3f}\nvali_mrr: {:3f}".format(self.train_mrr,self.vali_mrr))
        if hasattr(self.model, 'feature_importances_'):
            print(
                "Feature Importances:",
                sorted(
                    zip(feature_names, [i for i in self.model.feature_importances_]), key=lambda tup: tup[1], reverse=True,
                ),
            )
        else:
            print("Not explainable.")

    def dump(self,filename: str) -> None:
        joblib.dump(self.model, '{}.joblib'.format(filename))


def tune_lightgbm() -> ExperimentResult:
    experiments: List[ExperimentResult] = []
    qids_train = train.groupby("q_id")["q_id"].count().to_numpy()
    qids_vali = vali.groupby("q_id")["q_id"].count().to_numpy()

    for rnd in tqdm(range(n_rand)):
            params: Dict[str,str] = {
                "boosting_type": "gbdt",
                "objective": "lambdarank",
                "random_state": rnd,
            }
            m = lgb.LGBMRanker(**params)
            m.fit(
                X=X_train,
                y=y_train,
                group=qids_train,
                eval_set=[(X_vali, y_vali)],
                eval_group=[qids_vali],
                verbose=False,
                eval_at=10,
            )

            train_mrr = safe_mean(compute_clickout_RR(m,train))
            vali_mrr = safe_mean(compute_clickout_RR(m,vali))
            result = ExperimentResult(m,params,train_mrr,vali_mrr)
            experiments.append(result)

    return max(experiments, key = lambda tup: tup.vali_mrr)


# ok now evaluate the model on the metric that we care about: Mean Reciprocal Rank (MRR)
def calc_RR(sorted_item_list: List[bool]) -> float:
    """
    The way I've set this up: we want the True label (the item that the user really clicked)
    to appear at the first rank. Return the rank where it really appears
    """
    return 1 / (sorted_item_list.index(True) + 1)



def compute_clickout_RR(model: Any, data: pd.DataFrame) -> List[float]:
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
        qid_scores = model.predict(X_qid).ravel()

        # predict on features X and re-sort the items.. return a sorted list of item ids
        impressions_shuffled = [i[0] for i in sorted(zip(y_qid, qid_scores), key=lambda tup: tup[1],reverse=True)]
        rr = calc_RR(impressions_shuffled) # Score based upon the index where you found the relevant item.
        reciprocal_ranks.append(rr)
        # we don't quite care about the real choice in creating features here
        # because we're not evaluating click accuracy, we're using a listwise metric

    return reciprocal_ranks


MRR_train = safe_mean(compute_clickout_RR(f,train))
MRR_vali = safe_mean(compute_clickout_RR(f,vali))

print("RESULTS FOR USER BLIND DATA")
print("MRR_train: {:3f}\nMRR_vali: {:3f}\n".format(MRR_train,MRR_vali))

test_adv = test[test.is_advantaged_user == 1]
test_disadv = test[test.is_advantaged_user == 0]

# get the reciprocal ranks for all clickouts in each dataset
test_adv_ranks = compute_clickout_RR(f,test_adv)
test_disadv_ranks = compute_clickout_RR(f,test_disadv)

MRR_test_adv = safe_mean(test_adv_ranks)
MRR_test_disadv = safe_mean(test_disadv_ranks)
MRR_test_all = safe_mean(test_adv_ranks + test_disadv_ranks)
print("LR results on test set")
print("MRR_advantaged: {:3f}\nMRR_disadvantaged: {:3f}\nMRR_all: {:3f}\n".format(MRR_test_adv, MRR_test_disadv, MRR_test_all))


print("Training lightGBM")
l_gbm = tune_lightgbm()
l_gbm.outputs()
print("test MRR: {}".format(safe_mean(compute_clickout_RR(l_gbm.model,test))))
l_gbm.dump("forest")
# random forest listwise stats over the test set
test_adv_ranks_gbm = compute_clickout_RR(l_gbm.model,test_adv)
test_disadv_ranks_gbm = compute_clickout_RR(l_gbm.model,test_disadv)

# do this so I don't evaluate the whole test set separately from the subdivided
# test  set -- MRR takes a while to compute
MRR_test_adv_rf = safe_mean(test_adv_ranks_gbm)
MRR_test_disadv_rf = safe_mean(test_disadv_ranks_gbm)
MRR_test_all_rf = safe_mean(test_adv_ranks_gbm + test_disadv_ranks_gbm)

lgbm_results = {"all": MRR_test_all_rf, "adv": MRR_test_adv_rf, "dis": MRR_test_disadv_rf}
print("\nlightGBM TEST-SET RESULTS: \ntotal MRR: {all:3f}\nadvangaged MRR: {adv:3f}\ndisadvangaged MRR: {dis:3f}\n\n".format(**lgbm_results))
