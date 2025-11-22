# ============================================================
# Standard Libraries
# ============================================================
from datetime import date, timedelta, datetime
from typing import List

# ============================================================
# PySpark Libraries
# ============================================================
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

# ============================================================
# Configuration
# ============================================================
hdfs_namenode = 'hdfs://njbbepapa1.nss.vzwnet.com:9000'
base_dir = "/user/kovvuve/owl_history_v3/date="
TIME_COL = "time"

feature_groups = {
    "signal_quality": ["4GRSRP", "4GRSRQ", "SNR", "4GSignal", "BRSRP", "RSRQ", "5GSNR", "CQI"],
    "throughput_data": [
        "LTEPDSCHPeakThroughput", "LTEPDSCHThroughput",
        "LTEPUSCHPeakThroughput", "LTEPUSCHThroughput",
        "5GNRPDSCHThroughput", "5GNRPUSCHThroughput",
        "5GNRPDSCHPeakThroughput", "5GNRPUSCHPeakThroughput",
        "5GNRRxPDCPBytes", "5GNRPxTxPDCPBytes"
    ],
}
ALL_FEATURES = feature_groups["signal_quality"] + feature_groups["throughput_data"]
ZERO_LIST = ["RSRQ", "4GRSRQ", "4GRSRP", "BRSRP"]


#from preprocess_utils import convert_string_numerical, forward_fill
#from preprocess_utils import HourlyIncrementProcessor, split_into_subseries



def convert_string_numerical(
    df: DataFrame,
    cols_to_cast: List[str],
    decimal_places: int = 2
) -> DataFrame:
    """
    Casts selected columns to DoubleType and rounds them to a specified
    number of decimal places.

    Args:
        df: The input PySpark DataFrame.
        cols_to_cast: A list of column names to cast and round.
        decimal_places: The number of decimal places to round to. Defaults to 2.

    Returns:
        A new DataFrame with the specified columns cast and rounded.
    """
    for c in cols_to_cast:
        if c in df.columns:

            df = df.withColumn(
                c,
                F.round(F.col(c).cast(DoubleType()), decimal_places)
            )
    return df


class HourlyIncrementProcessor:
    """
    Steps:
      - 'hourly' : hourly averages (processed columns) + carry-through of ALL other columns via FIRST()
      - 'incr'   : replace processed features with smoothed increments
      - 'log'    : replace processed features with log1p(increments)
      - 'fill'   : forward-fill zeros in the (current) processed feature columns
    Notes:
      • All columns not listed in `columns` are preserved. During the 'hourly' aggregation,
        they are reduced with FIRST(ignorenulls=True) within each (partition_cols, time_col) group.
        Adjust that policy if you prefer MIN/MAX/LAST/etc.
    """

    def __init__(self, df: DataFrame, columns: List[str], partition_col=("sn",), time_col: str = TIME_COL):
        self.df = df
        self.columns = list(columns)
        self.partition_cols = list(partition_col) if isinstance(partition_col, (list, tuple)) else [partition_col]
        self.time_col = time_col

        # identify carry-through columns (everything except keys + processed columns)
        key_cols = set(self.partition_cols + [self.time_col])
        self.other_cols = [c for c in self.df.columns if c not in self.columns and c not in key_cols]

        self.df_hourly = None
        self._done = set()

    def compute_hourly_average(self):
        # averages for processed columns
        agg_proc = [F.round(F.avg(c), 2).alias(c) for c in self.columns]
        # carry-through for all remaining columns (FIRST non-null within the hour)
        agg_other = [F.first(col(c), ignorenulls=True).alias(c) for c in self.other_cols]

        self.df_hourly = (
            self.df
            .groupBy(*self.partition_cols, self.time_col)
            .agg(*agg_proc, *agg_other)
        )
        self._done.add("hourly")

    def compute_increments(self, partition_cols=None, order_col=None):
        if "hourly" not in self._done:
            self.compute_hourly_average()

        partition_cols = self.partition_cols if partition_cols is None else partition_cols
        order_col = self.time_col if order_col is None else order_col
        w = Window.partitionBy(*partition_cols).orderBy(order_col)

        for c in self.columns:
            prev = lag(col(c), 1).over(w)
            raw_incr = col(c) - prev
            prev_incr = lag(raw_incr, 1).over(w)

            incr = when(col(c) < prev, when(prev_incr.isNotNull(), prev_incr).otherwise(lit(0))).otherwise(raw_incr)
            prev_smooth = lag(incr, 1).over(w)
            smoothed = when(incr < 0, prev_smooth).otherwise(incr)
            self.df_hourly = self.df_hourly.withColumn(c, F.round(smoothed, 2))

        # drop rows where increments are null (first row per partition)
        self.df_hourly = self.df_hourly.na.drop(subset=self.columns)
        self._done.add("incr")

    def apply_log_transform(self):
        if "incr" not in self._done:
            self.compute_increments()

        for c in self.columns:
            self.df_hourly = self.df_hourly.withColumn(
                c, F.round(F.log1p(F.when(col(c) < 0, lit(0)).otherwise(col(c))), 2)
            )
        self._done.add("log")

    def fill_zero_with_previous(self, partition_cols=None, order_col=None):
        if "log" not in self._done and "incr" not in self._done and "hourly" not in self._done:
            self.compute_hourly_average()

        partition_cols = self.partition_cols if partition_cols is None else partition_cols
        order_col = self.time_col if order_col is None else order_col

        w_ffill = (
            Window.partitionBy(*partition_cols)
            .orderBy(order_col)
            .rowsBetween(Window.unboundedPreceding, 0)
        )

        for c in self.columns:
            tmp = f"__{c}_nz"
            self.df_hourly = (
                self.df_hourly
                .withColumn(tmp, when(col(c) != 0, col(c)))
                .withColumn(c, last(tmp, ignorenulls=True).over(w_ffill))
                .drop(tmp)
            )
        self._done.add("fill")

    def run(self, steps=("hourly", "incr", "log", "fill")):
        wanted = list(steps)
        for step in wanted:
            if step == "incr" and "hourly" not in self._done:
                self.compute_hourly_average()
            if step == "log" and "incr" not in self._done:
                self.compute_increments()
            if step == "fill" and not ({"hourly", "incr", "log"} & self._done):
                self.compute_hourly_average()

            if step == "hourly" and "hourly" not in self._done:
                self.compute_hourly_average()
            elif step == "incr" and "incr" not in self._done:
                self.compute_increments()
            elif step == "log" and "log" not in self._done:
                self.apply_log_transform()
            elif step == "fill" and "fill" not in self._done:
                self.fill_zero_with_previous()
        return self


def forward_fill(df: DataFrame, cols_to_process: List[str], partition_col: str, order_col: str) -> DataFrame:
    """
    Performs a forward fill on specified columns of a PySpark DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        cols_to_process (List[str]): A list of column names to apply forward fill.
        partition_col (str): The column to partition the window by (e.g., 'sn').
        order_col (str): The column to order the window by (e.g., 'time').

    Returns:
        DataFrame: The DataFrame with specified columns forward-filled.
    """
    window_spec = Window.partitionBy(partition_col).orderBy(order_col) \
                        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    # Calculate the mean of each column to use as a fallback value
    mean_values = df.select([F.mean(col_name).alias(f"mean_{col_name}") for col_name in cols_to_process]).first()
    for col_name in cols_to_process:
        # Step 1: Replace 0 with nulls
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) == 0, F.lit(None)).otherwise(F.col(col_name))
        )
        # Step 2: Forward fill the nulls
        df = df.withColumn(
            col_name,
            F.last(F.col(col_name), ignorenulls=True).over(window_spec)
        )
        # Step 3: Fill any remaining nulls (e.g., at the beginning of the partition) with the mean
        if mean_values is not None and mean_values[f"mean_{col_name}"] is not None:
            df = df.fillna({col_name: mean_values[f"mean_{col_name}"]})
    return df



if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName('NoveltyBatchPipeline')
        .config("spark.sql.adapative.enabled", "true")
        .config("spark.sql.shuffle.partitions", 1200)   
        .getOrCreate()
    )
# ============================================================
# Main Pipeline Function
# ============================================================

    # 1️⃣ Initialize Spark
    spark = (
        SparkSession.builder
        .appName("NoveltyBatchPipeline")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", 1200)
        .getOrCreate()
    )

    # 2️⃣ Generate file paths (last 200 hours)
    now = datetime.now()
    file_paths = [
        f"{hdfs_namenode}/{base_dir}{(now - timedelta(hours=i)).strftime('%Y-%m-%d')}/hr={(now - timedelta(hours=i)).strftime('%H')}"
        for i in range(200)
    ]

    # 3️⃣ Read raw data
    df = spark.read.option("header", "true").csv(file_paths)

    # 4️⃣ Basic cleaning & selection
    df = (
        df.withColumnRenamed("mdn", "MDN")
          .withColumn("MDN", F.regexp_replace(col("MDN"), '"', ''))
          .withColumn(TIME_COL, F.from_unixtime(F.col("ts") / 1000).cast("timestamp"))
          .select(["sn", "MDN", TIME_COL] + ALL_FEATURES)
          .dropDuplicates()
    )

    # 5️⃣ Cast numeric fields and forward fill missing/zero values
    df = convert_string_numerical(df, ALL_FEATURES)
    df = forward_fill(df, ZERO_LIST, partition_col="sn", order_col=TIME_COL)

    # 6️⃣ Throughput features → increments → log transform → zero-fill
    proc = (
        HourlyIncrementProcessor(df, feature_groups["throughput_data"], partition_col=["sn"])
        .run(steps=("incr", "log", "fill"))
    )
    df = proc.df_hourly

    # 7️⃣ Convert into model training samples (sequence windows)
    df_slice = (
        split_into_subseries(df, length=200, shift=1, sn_col="sn", time_col=TIME_COL)
        .drop("sn")
        .withColumnRenamed("series_id", "sn")
    )

    # 8️⃣ Save final model-ready dataset
    output_path = "/user/ZheS/owl_anomaly/processed_ask-ncm1100_hourly_features/data"
    df_slice.write.mode("overwrite").parquet(output_path)

    print(f"🚀 Preprocessing complete. Output saved to:\n{output_path}")
