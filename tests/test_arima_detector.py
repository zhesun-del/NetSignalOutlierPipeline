import sys
import os

# =========================================
# Dynamically add /src to sys.path
# =========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from modeling.detectors.arima_detector import ARIMAAnomalyDetectorFuture

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow
import matplotlib.pyplot as plt


# =========================================
# Setup MLflow Tracking
# =========================================
mlflow.set_tracking_uri("http://njbbvmaspd11:5001")
mlflow.set_experiment("NetSignal_ARIMAForecast_NoveltyTest")


# =========================================
# Synthetic Dataset Generator
# =========================================
def generate_synthetic_arima_data():
    """
    Generate synthetic time series with clear seasonality and injected anomalies
    for future-forecast-based novelty detection.
    """
    n_points = 200
    timestamps = pd.date_range(datetime.now() - timedelta(hours=n_points), periods=n_points, freq="H")

    # Base sinusoidal with noise
    values = 2 * np.sin(np.linspace(0, 4 * np.pi, n_points)) + np.random.normal(0, 0.2, n_points)

    df = pd.DataFrame({
        "sn": ["TEST_SN_1"] * n_points,
        "time": timestamps,
        "throughput": values
    })

    # Inject anomalies ONLY in the future forecast horizon (last 24 points)
    df.loc[n_points - 20:n_points - 10, "throughput"] += 4   # high spikes
    df.loc[n_points - 10:, "throughput"] -= 4                # deep drops

    return df


# =========================================
# Test ARIMA Future Forecast-Based Novelty Detection
# =========================================
def test_arima_novelty():
    df = generate_synthetic_arima_data()

    split_idx = 20  # Number of steps to forecast (future horizon)

    with mlflow.start_run(run_name="ARIMA_Forecast_NoveltyDetection_Test"):
        params = {
            "season_length": 24,
            "confidence_level": 95,
            "freq": "H",
            #"split_idx": split_idx,
            "anomaly_direction": "both",
        }
        mlflow.log_params(params)

        detector = ARIMAAnomalyDetectorFuture(
            df=df,
            time_col="time",
            feature="throughput",
            season_length=params["season_length"],
            confidence_level=params["confidence_level"],
            freq=params["freq"],
            anomaly_direction=params["anomaly_direction"],
            split_idx=params["split_idx"],
        )

        detector.run()

        fig = detector.plot_anomalies()
        mlflow.log_figure(fig, "arima_forecast_novelty_detection.png")

        mlflow.log_metric("n_train_samples", len(df) - split_idx)
        mlflow.log_metric("n_test_samples", split_idx)
        mlflow.log_metric("n_anomalies_detected", detector.result_df["anomaly"].sum())

        print("\n📌 ARIMA Future Forecast Novelty Detection Results:")
        print(detector.result_df[detector.result_df["anomaly"]])

    return detector.result_df


# =========================================
# Entry Point
# =========================================
if __name__ == "__main__":
    print("\n========== Running ARIMA Future Forecast Novelty Detection Test ==========")
    results = test_arima_novelty()
    print("\n🎯 ARIMA Future Detection Testing Completed.")
