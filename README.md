# Trivago Recsys Experiments


## How to use:
1. These scripts expect a data/trivago subdirectory...
    - mkdir data
    - put the trivago dataset in data/ OR if using ada, create a link to the trivago folder in the storage node.
    - the files you need are: ~train.csv~, ~validation.csv~, ~confirmation.csv~

2. compute features for the experiments:
    - python3 load_trivago.py
    - python3 hash_user_id.py
    - python3 load_trivago_blind.py


3. run the experiments:
    - python3 train_eval.py
    - python3 train_eval_blind.py


## More details on each script:

### load_trivago.py
This script creates a readable parquet file which contains all the data which is required to train the models. In this file, I've included a range of dataclasses to help me understand the dataset, and to make feature extraction more intuitive and readable. The objects I've defined are as follows:
1. Hotel
2. Interaction
3. Session
4. UserProfile
5. SessionData

Following LogicAI's 2019 recsys strategy, I computed user features on a rolling basis to prevent overfitting. Particularly, I sorted sessions by their starting timestamp, and added each interaction to a user profile and graph only after features had been extracted from that interaction.

### compute_ctr.py


### extract_hotel_features.py

### hash_user_id.py


### load_trivago_blind.py
This script would remarkably similar to the above, but it excludes user profiles and computes fewer features. It also constructs a user-item graph as a ~session-item~ graph instead, which is more sparse.



## Future work/stuff I didn't cover this summer.
I tackled a limited range of features which were considered most important in the challenge, while using my best judgement to exclude ones which I considered "cheating". For example, the first-place submission had

Some more features may have been helpful in boosting our final test MRR (.576)

In these experiments, I found that the user was really not important in recommending items over such a short period. That is, people likely do not plan multiple, different vacations/work trips in one week.
