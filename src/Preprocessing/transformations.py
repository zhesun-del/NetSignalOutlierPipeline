# ============================================================
# PySpark Libraries
# ============================================================
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

def unpivot_wide_to_long(df, time_col, feature_cols):
    # Build a stack(expr) for unpivot: (feature, value)
    n = len(feature_cols)
    expr = "stack({n}, {pairs}) as (feature, value)".format(
        n=n,
        pairs=", ".join([f"'{c}', `{c}`" for c in feature_cols])
    )
    return df.select("sn", time_col, F.expr(expr))


def split_into_subseries(
    df: DataFrame,
    length: int,
    shift: int,
    sn_col: str = "sn",
    time_col: str = "time",
    id_col: str = "series_id",
    time_fmt: str = "yyyyMMddHHmmss"
) -> DataFrame:
    """
    Slice each SN's time series into overlapping windows of `length` rows,
    advancing by `shift` rows, ordered by `time_col`.

    Returns the original rows plus a new `id_col` that is the concatenation of
    sn and the sub-series start time formatted with `time_fmt`.
    """
    assert length > 0 and shift > 0, "length and shift must be positive integers"

    # 1) Order within each sn and assign row numbers
    w_rn = Window.partitionBy(sn_col).orderBy(time_col)
    df_rn = df.withColumn("rn", F.row_number().over(w_rn))

    # 2) For each sn, compute max rn and generate valid window starts: 1, 1+shift, ..., <= N-length+1
    max_rn = df_rn.groupBy(sn_col).agg(F.max("rn").alias("N"))
    starts = (
        max_rn
        .withColumn(
            "start_rn",
            F.when(
                F.col("N") >= F.lit(length),
                F.expr(f"sequence(1, N - {length} + 1, {shift})")  # array of start indices
            )
        )
        .select(sn_col, F.explode("start_rn").alias("start_rn"))  # one row per window start
    )

    # 3) Assign rows to windows where rn ∈ [start_rn, start_rn + length - 1]
    df_windows = (
        df_rn.join(starts, on=sn_col, how="inner")
             .filter((F.col("rn") >= F.col("start_rn")) & (F.col("rn") < F.col("start_rn") + length))
    )

    # 4) Compute sub-series start time per (sn, start_rn) and build series_id
    w_win = Window.partitionBy(sn_col, "start_rn")
    df_windows = df_windows.withColumn("series_start_time", F.min(time_col).over(w_win))
    df_windows = df_windows.withColumn(
        id_col,
        F.concat_ws("_", F.col(sn_col), F.date_format(F.col("series_start_time"), time_fmt))
    )

    # 5) Return rows annotated with the sub-series id (drop helpers)
    return df_windows.drop("rn", "N", "start_rn", "series_start_time")

