from ..detectors.arima_detector import ARIMAAnomalyDetectorFuture



from functools import reduce
from typing import List, Tuple
# =============================
# PySpark Libraries
# =============================
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, lag, last, lit, row_number, when
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    BooleanType, DoubleType, FloatType, StringType,
    StructField, StructType, TimestampType
)

# =============================
# Third-Party Libraries
# =============================
import numpy as np
import pandas as pd

# =============================

arima_schema = StructType([
    StructField("sn", StringType(), True),
    StructField("time", TimestampType(), True),
    StructField("feature", StringType(), True),   # <── added
    StructField("value", FloatType(), True),
    StructField("is_outlier", BooleanType(), True),
    StructField("forecast", FloatType(), True),
    StructField("lo", FloatType(), True),
    StructField("hi", FloatType(), True),
])


def groupwise_novelty_autoarima(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ARIMAAnomalyDetectorFuture per (sn, feature) group.
    Returns standardized columns: sn, time, feature, value, is_outlier, forecast, lo, hi.
    """
    import numpy as np
    if not hasattr(np, "round_"):
        np.round_ = np.round   # patch for statsforecast

    columns = list(arima_schema.names)

    pdf = pdf.copy()
    pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
    pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
    pdf = pdf.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    if len(pdf) < 10:
        return pd.DataFrame(columns=columns)

    try:
        from pandas.core.strings.accessor import StringMethods
        pd.core.strings.StringMethods = StringMethods
        det = ARIMAAnomalyDetectorFuture(
            df=pdf,
            time_col="time",
            feature="value",
            season_length=1,
            confidence_level=99,
            freq="h",
            anomaly_direction="lower",
            split_idx=1,
            unique_id=pdf["sn"].iloc[0],
        )
        det.run()
        result_df = det.result_df.copy()
        if result_df.empty:
            return pd.DataFrame(columns=columns)

        lo_col = f"AutoARIMA-lo-{det.confidence_level}"
        hi_col = f"AutoARIMA-hi-{det.confidence_level}"
        result_df = result_df.rename(columns={
            "unique_id": "sn",
            "ds": "time",
            "y": "value",
            "anomaly": "is_outlier",
            "AutoARIMA": "forecast",
            lo_col: "lo",
            hi_col: "hi",
        })

        # Keep only anomalies
        result_df = result_df[result_df["is_outlier"]].copy()
        if result_df.empty:
            return pd.DataFrame(columns=columns)

        result_df["feature"] = pdf["feature"].iloc[0]
        result_df = result_df[columns]

        result_df["value"] = result_df["value"].astype(float)
        result_df["forecast"] = result_df["forecast"].astype(float)
        result_df["lo"] = result_df["lo"].astype(float)
        result_df["hi"] = result_df["hi"].astype(float)
        result_df["is_outlier"] = result_df["is_outlier"].astype(bool)

        return result_df

    except Exception:
        return pd.DataFrame(columns=columns)

