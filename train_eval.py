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
n_rand = 2

data_all = pd.read_parquet("data/trivago/data_all.parquet", engine="pyarrow")
print("loaded data",flush=True)
# split the data back into three frames
train = data_all[data_all.grp == 0]
vali = data_all[data_all.grp == 1]
test = data_all[data_all.grp == 2]
print("split data",flush=True)
feature_names = set(data_all.columns) - set(["y", "q_id", "grp", "choice_idx"])


# second on a specific subsection of the df. we want a transfored version of all rows corresppondding with
# a single q_id, and its listwise label for computing MRR
def get_qid_data(df: pd.DataFrame, q_id: str, scaler: StandardScaler) -> Tuple[np.ndarray,np.ndarray]:
    """
    Helps me compute mean reciprocal rank later on, but without re-computing features.
    (Unfortunately I stil need to convert to a dict and then use DictVectorizer and sclaer on the dict).
    """
    # find the piece of the dataframe that i was talking about
    slice = df[df.q_id == q_id]
    # get features and labels into numpy format -- transform with standard scaler
    X_slice = fscale.transform(slice[feature_names])
    y_slice = np.array(slice["y"])
    return X_slice, y_slice # pack and return them

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

"""
test_pred = f.predict_proba(X_test)[:, 1].ravel()
test_auc = roc_auc_score(y_true=y_test, y_score=test_pred)
"""

# how well did my model learn the data
print("\n---Data Shape---")
print("train shape: {}\n vali shape {}\n test shape {}\n".format(str(X_train.shape),str(X_vali.shape),str(X_test.shape)))
print("train AUC: {:3f}\n vali AUC {:3f}".format(train_auc,vali_auc))

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
        print("Model params",self.params)
        print("Results\n-------\n train_auc: {:3f}\nmrr_train: {:3f}\n vali_auc: {:3f}\n vali_mrr: {:3f}".format(self.train_auc,self.mrr_train,self.vali_auc,self.mrr_vali))
        if hasattr(self.model, 'feature_importances_'):
            print(
                "Feature Importances:",
                sorted(
                    zip(feature_names, self.model.feature_importances_), key=lambda tup: tup[1], reverse=True,
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
    unique_query_ids: List[str] = list(data.q_id.unique())
    reciprocal_ranks: List[float] = []
    for query_str in unique_query_ids:  # 1. get all the clickout ids, which should be of the form session_id/step

        X_qid, y_qid = get_qid_data(data, query_str, fscale)

        y_qid = y_qid.ravel()
        # extract features takes a session, a step, and a choice index.
        if True not in y_qid:
            continue
        #assert(list(y_qid).index(True) == o.impressions.index(o.action_on))

        # this outputs two values -- we want the probability of the second class ( found at index 1)
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
print("MRR_train: {:3f}\nMRR_vali: {:3f}".format(MRR_train,MRR_vali))

print("trying a bunch of RF models")
rf = tune_RF_model()
rf.outputs()

print("and some MLP classifiers")
nn = tune_MLP_model()
nn.outputs()


test_mrr_rf = safe_mean(compute_clickout_RR(rf.model,test))
test_mrr_nn = safe_mean(compute_clickout_RR(nn.model,test))

print("EXPERIMENTS:\nrf test_MRR: {:3f}\nnn test_MRR: {:3f}".format(test_mrr_rf,test_mrr_nn),flush=True)
