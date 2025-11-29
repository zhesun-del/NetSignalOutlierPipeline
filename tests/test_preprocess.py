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
    # 2. Normalization
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
    
    # ---------------------------------------
    # 3. Transformation/Structuring Pipeline
    # ---------------------------------------

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


"""
:: loading settings :: url = jar:file:/usr/apps/vmas/spark/jars/ivy-2.5.1.jar!/org/apache/ivy/core/settings/ivysettings.xml

=== Starting Test Preprocessing Job ===
Start Time: 2025-11-22 18:17:56.458624

Data Source Paths (24 total):
  hdfs://njbbepapa1.nss.vzwnet.com:9000//user/kovvuve/owl_history_v3/date=2025-11-22/hr=18
  hdfs://njbbepapa1.nss.vzwnet.com:9000//user/kovvuve/owl_history_v3/date=2025-11-22/hr=17
  hdfs://njbbepapa1.nss.vzwnet.com:9000//user/kovvuve/owl_history_v3/date=2025-11-22/hr=16
  ...

Initial Raw DataFrame:
+-----------+-----------+-------------------+------+------+----+--------+------+-----+-----+---+-----------+-----------+------------------+--------------+-------------------+---------------+
|sn         |MDN        |time               |4GRSRP|4GRSRQ|SNR |4GSignal|BRSRP |RSRQ |5GSNR|CQI|TxPDCPBytes|RxPDCPBytes|TotalBytesReceived|TotalBytesSent|TotalPacketReceived|TotalPacketSent|
+-----------+-----------+-------------------+------+------+----+--------+------+-----+-----+---+-----------+-----------+------------------+--------------+-------------------+---------------+
|ACR50401575|19413934329|2025-11-21 23:17:18|-97.5 |-12   |5.0 |-85.0   |-107.2|-11.2|14.8 |12 |0          |0          |100250181255      |31463170162   |86985569           |49587605       |
|ACR50401575|19413934329|2025-11-21 23:43:47|-98   |-12.5 |2.5 |-84.8   |-106  |-11.2|16.2 |4  |0          |0          |100648440635      |31474127807   |87305781           |49625077       |
|ACR51701149|17302270048|2025-11-21 23:29:14|0     |0     |null|0.0     |-83.5 |-11  |28   |13 |0          |0          |100671223226      |8698073305    |78282984           |18446912       |
|ACR50401575|19413934329|2025-11-22 00:20:51|-98   |-10.8 |3.8 |-86.8   |-107.8|-11.2|13.8 |6  |0          |0          |100807856664      |31502919167   |87475637           |49720510       |
|ACR50401575|19413934329|2025-11-22 01:45:00|-98   |-11.2 |3.2 |-86.2   |-106.8|-11.2|15.2 |13 |0          |0          |101540833870      |31540529219   |88085415           |49841686       |
+-----------+-----------+-------------------+------+------+----+--------+------+-----+-----+---+-----------+-----------+------------------+--------------+-------------------+---------------+
only showing top 5 rows


Processed Data Sample:
+-----------+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+
|sn         |time               |TxPDCPBytes|RxPDCPBytes|TotalBytesReceived|TotalBytesSent|TotalPacketReceived|TotalPacketSent|MDN        |4GRSRP             |4GRSRQ            |SNR |4GSignal|BRSRP|RSRQ |5GSNR|CQI |
+-----------+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+
|ACR42006080|2025-11-21 19:05:18|null       |null       |11.5              |11.75         |6.48               |6.75           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-98.0|-11.2|10.8 |10.0|
|ACR42006080|2025-11-21 19:10:33|null       |null       |11.35             |11.75         |6.44               |6.74           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-98.0|-12.0|9.0  |14.0|
|ACR42006080|2025-11-21 19:15:50|null       |null       |14.01             |13.02         |7.42               |7.3            |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-94.8|-11.2|14.8 |9.0 |
|ACR42006080|2025-11-21 19:21:05|null       |null       |11.66             |11.75         |6.51               |6.75           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-97.0|-11.5|9.8  |14.0|
|ACR42006080|2025-11-21 19:26:21|null       |null       |11.45             |12.25         |6.47               |6.81           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-95.8|-11.5|12.5 |7.0 |
+-----------+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+
only showing top 5 rows


=== Test Processing Completed ===

=== 3. Testing Wide → Long Transformation ===
+-----------+-------------------+-----------+-------------------+
|sn         |time               |feature    |value              |
+-----------+-------------------+-----------+-------------------+
|ACR42006080|2025-11-21 19:05:18|4GRSRP     |-55.639322157434435|
|ACR42006080|2025-11-21 19:05:18|4GRSRQ     |-5.658491253644314 |
|ACR42006080|2025-11-21 19:05:18|SNR        |null               |
|ACR42006080|2025-11-21 19:05:18|4GSignal   |0.0                |
|ACR42006080|2025-11-21 19:05:18|BRSRP      |-98.0              |
|ACR42006080|2025-11-21 19:05:18|RSRQ       |-11.2              |
|ACR42006080|2025-11-21 19:05:18|5GSNR      |10.8               |
|ACR42006080|2025-11-21 19:05:18|CQI        |10.0               |
|ACR42006080|2025-11-21 19:05:18|TxPDCPBytes|null               |
|ACR42006080|2025-11-21 19:05:18|RxPDCPBytes|null               |
+-----------+-------------------+-----------+-------------------+
only showing top 10 rows


=== 4. Testing Subseries Splitting ===
+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+--------------------------+
|time               |TxPDCPBytes|RxPDCPBytes|TotalBytesReceived|TotalBytesSent|TotalPacketReceived|TotalPacketSent|MDN        |4GRSRP             |4GRSRQ            |SNR |4GSignal|BRSRP|RSRQ |5GSNR|CQI |series_id                 |
+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+--------------------------+
|2025-11-21 19:05:18|null       |null       |11.5              |11.75         |6.48               |6.75           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-98.0|-11.2|10.8 |10.0|ACR42006080_20251121190518|
|2025-11-21 19:10:33|null       |null       |11.35             |11.75         |6.44               |6.74           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-98.0|-12.0|9.0  |14.0|ACR42006080_20251121190518|
|2025-11-21 19:15:50|null       |null       |14.01             |13.02         |7.42               |7.3            |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-94.8|-11.2|14.8 |9.0 |ACR42006080_20251121190518|
|2025-11-21 19:21:05|null       |null       |11.66             |11.75         |6.51               |6.75           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-97.0|-11.5|9.8  |14.0|ACR42006080_20251121190518|
|2025-11-21 19:26:21|null       |null       |11.45             |12.25         |6.47               |6.81           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-95.8|-11.5|12.5 |7.0 |ACR42006080_20251121190518|
|2025-11-21 19:31:37|null       |null       |11.56             |12.23         |6.49               |6.83           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-97.5|-11.5|10.2 |14.0|ACR42006080_20251121190518|
|2025-11-21 19:36:53|null       |null       |12.83             |12.32         |6.86               |7.04           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-94.0|-11.2|12.5 |5.0 |ACR42006080_20251121190518|
|2025-11-21 19:42:09|null       |null       |11.46             |11.78         |6.52               |6.77           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-95.0|-11.2|14.5 |13.0|ACR42006080_20251121190518|
|2025-11-21 19:47:26|null       |null       |11.17             |11.61         |6.36               |6.66           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-94.8|-11.2|10.0 |6.0 |ACR42006080_20251121190518|
|2025-11-21 19:52:41|null       |null       |11.4              |11.81         |6.48               |6.84           |12088217415|-55.639322157434435|-5.658491253644314|null|0.0     |-96.2|-11.5|9.2  |15.0|ACR42006080_20251121190518|
+-------------------+-----------+-----------+------------------+--------------+-------------------+---------------+-----------+-------------------+------------------+----+--------+-----+-----+-----+----+--------------------------+
only showing top 10 rows


=== End-to-End Functional Test Completed ===
End Time: 2025-11-22 18:22:36.412663

"""
