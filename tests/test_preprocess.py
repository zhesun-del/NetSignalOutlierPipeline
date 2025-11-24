import sys
import os

# Dynamically add /src to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_PATH)

from Preprocessing.cleaning import convert_string_numerical, forward_fill,HourlyIncrementProcessor
from Preprocessing.transformations import unpivot_wide_to_long, split_into_subseries
from config.feature_config import (
    HDFS_NAMENODE,
    BASE_DIR,
    TIME_COL,
    feature_groups,
    ALL_FEATURES,
    ZERO_LIST,
    LOOKBACK_HOURS
)


from pyspark.sql import SparkSession, DataFrame
from datetime import datetime, timedelta    
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, when, lit, last


# ============================================================
# Spark Session Initialization
# ============================================================

def create_spark(app_name="NoveltyBatchPipelineTest"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", 800)
        .getOrCreate()
    )

# ============================================================
# Test Execution
# ============================================================

def run_preprocessing_test():
    spark = create_spark()

    print("\n=== Starting Test Preprocessing Job ===")
    print("Start Time:", datetime.now())

    # ---------------------------------------
    # Define HDFS input paths (last 24 hours)
    # ---------------------------------------
    file_paths = []
    now = datetime.now()

    for i in range(LOOKBACK_HOURS):
        dt = now - timedelta(hours=i)
        path = f"{HDFS_NAMENODE}/{BASE_DIR}{dt.strftime('%Y-%m-%d')}/hr={dt.strftime('%H')}"
        file_paths.append(path)

    print(f"\nData Source Paths ({len(file_paths)} total):")
    for p in file_paths[:3]:
        print(" ", p)
    print("  ...")

    # ---------------------------------------
    # Load raw feature DataFrame
    # ---------------------------------------
    sn_list = ['ACR50220495', 'ACR50219952', 'ACR45123236', 'ACR50709744', 'ACR45127066', 'ACR50407638', 'ACR51109908', 'ACR52417251', 'ACR51317239', 'ACR44810858', 'ACR43301951', 'ACR43103903', 'ACR43105974', 'ACR44214489', 'ACR52212239', 'ACR44717227', 'ACR50111657', 'ACR51112474', 'ACR44000230', 'ACR52505377', 'ACR45011967', 'ACR50210814', 'ACR43712925', 'ACR44700139', 'ACR50401575', 'ACR51312404', 'ACR52605358', 'ACR50204281', 'ACR44713139', 'ACR52304552', 'ACR50705978', 'ACR44510528', 'ACR43714196', 'ACR44909542', 'ACR52301175', 'ACR44406975', 'ACR44518289', 'ACR43403518', 'ACR44902646', 'ACR44003303', 'ACR51110264', 'ACR45105556', 'ACR42006080', 'ACR52601816', 'ACR44700010', 'ACR51519291', 'ACR51701149', 'ACR43513827', 'ACR50204843', 'ACR42812887', 'ACR44700266', 'ACR50719917', 'ACR43100493', 'ACR51106604', 'ACR43310012', 'ACR51505149', 'ACR50423435', 'ACR50906565', 'ACR43109313', 'ACR44723610', 'ACR51717554', 'ACR43308279', 'ACR44715171', 'ACR45004304', 'ACR44522300', 'ACR45125537', 'ACR51314147', 'ACR44902044', 'ACR50419211', 'ACR43400537', 'ACR51508875', 'ACR50907524', 'ACR42802896', 'ACR43103268', 'ACR44516105', 'ACR44801791', 'ACR50211956', 'ACR42807055', 'ACR45122687', 'ACR51304508', 'ACR44810561', 'ACR44007959', 'ACR43511767', 'ACR45100534', 'ACR45120057', 'ACR44902278', 'ACR51315781', 'ACR42407111', 'ACR50709571', 'ACR50205333', 'ACR44810509', 'ACR50115055', 'ACR50706528', 'ACR44005591', 'ACR44701895', 'ACR50208010', 'ACR42201924', 'ACR44010952', 'ACR51506275', 'ACR44900466']
    df = (
        spark.read.option("header", "true").csv(file_paths)
        .withColumnRenamed("mdn", "MDN")
        .withColumn("MDN", F.regexp_replace("MDN", '"', ''))
        .withColumn("time", F.from_unixtime(F.col("ts") / 1000.0).cast("timestamp"))
        .select(["sn", "MDN", "time"] + ALL_FEATURES)
        .dropDuplicates()
        .filter(F.col("sn").isin(sn_list))
        .filter(F.col("ModelName") == "ASK-NCM1100")   # optional filter
    )

    print("\nInitial Raw DataFrame:")
    df.show(5, truncate=False)

    # ---------------------------------------
    # 1. Type Conversion & Normalization
    # ---------------------------------------
    df = convert_string_numerical(df, ALL_FEATURES)

    df = forward_fill(df, ZERO_LIST, partition_col="sn", order_col="time")
    df = df.orderBy("sn", "time")

    # ---------------------------------------
    # 2. Increment & Transformation Pipeline
    # ---------------------------------------
    proc = HourlyIncrementProcessor(
        df,
        columns=feature_groups["throughput_data"],
        partition_col=["sn"]
    ).run(steps=('incr', 'log', 'fill'))

    df_processed = proc.df_hourly
    print("\nProcessed Data Sample:")
    df_processed.show(5, truncate=False)

    print("\n=== Test Processing Completed ===")


    print("\n=== 3. Testing Wide → Long Transformation ===")
    df_long = unpivot_wide_to_long(
        df_processed,
        time_col=TIME_COL,
        feature_cols=ALL_FEATURES
    )
    df_long.show(10, truncate=False)

    print("\n=== 4. Testing Subseries Splitting ===")
    df_slice = split_into_subseries(
        df_processed,
        length=200,
        shift=1,
        sn_col="sn",
        time_col=TIME_COL
    ).drop("sn")

    df_slice.show(10, truncate=False)

    print("\n=== End-to-End Functional Test Completed ===")
    print("End Time:", datetime.now())

    return df_processed

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":

    final_df = run_preprocessing_test()
