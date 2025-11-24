import sys
import os
import pandas as pd
import numpy as np
import mlflow
mlflow.set_tracking_uri("http://njbbvmaspd11:5001")
mlflow.set_experiment("NetSignal_EWMA_DetectionTest")

# =========================================
# Dynamically add /src to sys.path
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

# =========================================
# Import Custom Modules
# =========================================

from Preprocessing.cleaning import convert_string_numerical, forward_fill,HourlyIncrementProcessor
from Preprocessing.transformations import unpivot_wide_to_long
from config.feature_config import (
    HDFS_NAMENODE,
    BASE_DIR,
    TIME_COL,
    feature_groups,
    ALL_FEATURES,
    ZERO_LIST,
    PARAMS_KDE_EWMA
)
from modeling.detectors.ewma_detector import EWMAAnomalyDetector
from modeling.detectors.kde_detector import FeaturewiseKDENoveltyDetector
from config.feature_config import (
    PARAMS_KDE_EWMA
)
from modeling.distributed_detection_runtime.kde_ewma_runtime import (
    mlflow_tracked_kde_ewma_ensemble,
    both_schema
)

# =========================================
# Import PySpark 
# =========================================
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime, timedelta    
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, when, lit, last


if __name__ == "__main__":
    # Spark Session Initialization
    spark = SparkSession.builder \
        .appName("NetSignalOutlierPipelineTest") \
        .getOrCreate()

    df_slice=spark.read.parquet("/user/ZheS/owl_anomaly/processed_ask-ncm1100_hourly_features/data")\

    feature_cols = ['4GRSRP', '4GRSRQ', 'SNR', '4GSignal', 'BRSRP', 'RSRQ', '5GSNR', 'CQI']
    df_long = unpivot_wide_to_long(df_slice, time_col=TIME_COL, feature_cols=feature_cols)
    df_long.show(5, truncate=False)
    df_result = (
        df_long
        .groupBy("sn", "feature")
        .applyInPandas(mlflow_tracked_kde_ewma_ensemble, schema=both_schema)
    )
    df_result.show(10, truncate=False)