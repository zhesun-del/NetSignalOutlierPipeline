import os
import sys
import mlflow
import pandas as pd 

# =========================================
# Dynamically add /src to sys.path
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from modeling.detectors.ewma_detector import EWMAAnomalyDetector
from modeling.detectors.kde_detector import FeaturewiseKDENoveltyDetector
from config.feature_config import (
    PARAMS_KDE_EWMA
)
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, FloatType, BooleanType


both_schema = StructType([
    StructField("sn", StringType(), True),
    StructField("time", TimestampType(), True),
    StructField("feature", StringType(), True),
    StructField("value", FloatType(), True),
    StructField("is_outlier_kde", BooleanType(), True),
    StructField("is_outlier_ewma", BooleanType(), True),
])

def mlflow_tracked_kde_ewma_ensemble(pdf: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Apply KDE and EWMA anomaly detectors on a single (sn, feature) group.
    Logs metrics and params to MLflow.
    Returns only anomaly rows.
    """
    import mlflow
    mlflow.set_tracking_uri("http://njbbvmaspd11:5001")
    mlflow.set_experiment("Novelty_Detection")

    if pdf.empty:
        return pd.DataFrame(columns=["sn","time","feature","value","is_outlier_kde","is_outlier_ewma"])

    # Ensure types and ordering
    pdf = pdf.copy()
    pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
    pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
    pdf = pdf.sort_values("time").reset_index(drop=True)

    sn_id = pdf["sn"].iloc[0]
    feature_id = pdf["feature"].iloc[0]

    # Start MLflow nested run
    with mlflow.start_run(nested=True, run_name=f"Novelty-{sn_id}-{feature_id}"):
        try:
            # Log params if provided
            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # Minimum data check
            mlflow.log_metric("n_points", len(pdf))
            if len(pdf) < params.get("min_training_points", 10):
                mlflow.log_metric("anomalies_detected", 0)
                return pd.DataFrame(columns=["sn","time","feature","value","is_outlier_kde","is_outlier_ewma"])

            # ===== KDE Detector =====
            kde = FeaturewiseKDENoveltyDetector(
                df=pdf,
                feature_col="value",
                time_col="time",
                train_idx="all",
                new_idx=slice(-1, None),
                filter_percentile=params.get("filter_percentile", 99),
                threshold_percentile=params.get("threshold_percentile", 99),
                anomaly_direction=params.get("anomaly_direction", "low"),
            )
            out_kde = kde.fit()[["sn","time","value","is_outlier"]].rename(
                columns={"is_outlier": "is_outlier_kde"}
            )

            # ===== EWMA Detector =====
            ewma = EWMAAnomalyDetector(
                df=pdf,
                feature="value",
                timestamp_col="time",
                recent_window_size=params.get("recent_window_size", 1),
                window=params.get("window", 100),
                no_of_stds=params.get("no_of_stds", 3.0),
                n_shift=params.get("n_shift", 1),
                anomaly_direction=params.get("anomaly_direction", "low"),
                scaler=params.get("scaler", None),
                min_std_ratio=params.get("min_std_ratio", 0.01),
                use_weighted_std=params.get("use_weighted_std", False),
            )
            out_ewma = ewma.fit()[["sn","time","value","is_outlier"]].rename(
                columns={"is_outlier": "is_outlier_ewma"}
            )

            # ===== Join outputs =====
            base = pdf[["sn","time","feature","value"]]
            out = (
                base.merge(out_kde, on=["sn","time","value"], how="left")
                    .merge(out_ewma, on=["sn","time","value"], how="left")
            )

            out[["is_outlier_kde","is_outlier_ewma"]] = out[["is_outlier_kde","is_outlier_ewma"]].fillna(False)

            # Filter only anomalies
            out_anomalies = out[(out["is_outlier_kde"]) | (out["is_outlier_ewma"])]

            mlflow.log_metric("anomalies_detected", len(out_anomalies))
            return out_anomalies[["sn","time","feature","value","is_outlier_kde","is_outlier_ewma"]]

        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.log_metric("anomalies_detected", 0)
            return pd.DataFrame(columns=["sn","time","feature","value","is_outlier_kde","is_outlier_ewma"])

