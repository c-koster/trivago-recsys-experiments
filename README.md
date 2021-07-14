# Trivago Recsys Experiments

- expects a data/trivago subdirectory
- mkdir data
- then put the trivago dataset in data OR if using ada, create a link to the trivago folder in the storage node.


## load_trivago.py
This script should be run first, as it creates a readable parquet file which contains all the data which is required to train the models. In this file, I've included a range of dataclasses to help me understand the dataset, and to make feature extraction more intuitive and readable. The objects I've defined are as follows:
1. Hotel
2. Interaction
3. Session
4. UserProfile
5. SessionData
