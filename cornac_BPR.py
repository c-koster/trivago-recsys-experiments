"""
Collaborative filtering: Use user-item interactions to help in recommendations
"""

# boilerplate imports
import sys
import pyspark
from pyspark.ml.recommendation import ALS  # for Matrix Factorization using ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType

from reco_utils.common.timer import Timer
from reco_utils.common.notebook_utils import is_jupyter
from reco_utils.dataset.spark_splitters import spark_random_split
from reco_utils.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from reco_utils.common.spark_utils import start_or_get_spark


TOP_K = 25


# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap
spark = start_or_get_spark("ALS PySpark", memory="8g")

# get me a schema
# Note: The DataFrame-based API for ALS currently only supports integers for user and item ids.
# user_id,session_id,timestamp,step,action_type,reference,platform,city,device,current_filters,impressions,prices

schema = StructType(
    (
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("timestamp", LongType()),
        StructField("step", IntegerType()),
        StructField("action_type", StringType()),
        StructField("reference", IntegerType()),
    )
)
data = spark.read.csv('data/trivago/train.csv', header=True, schema=schema)
data.show()

# next transform data so that postive interactions are a 1 and non interactions
# are a -1


header = {
    "userCol": "user_id",
    "itemCol": "reference",
    "ratingCol": "action_type",
}


als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
)
