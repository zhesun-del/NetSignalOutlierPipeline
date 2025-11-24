import sys
import os

# Dynamically add /src to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_PATH)
from modeling.detectors.kde_detector import FeaturewiseKDENoveltyDetector


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


import mlflow
import os
import matplotlib.pyplot as plt




# ============================================================
# Setup MLflow Tracking
# ============================================================
mlflow.set_tracking_uri("http://njbbvmaspd11:5001")
mlflow.set_experiment("NetSignal_KDE_AnomalyTest")


# ============================================================
# Synthetic Dataset Generator
# ============================================================
def generate_synthetic_data(for_novelty=True):
    """
    Generate synthetic dataframe with injected anomalies.
    - If for_novelty=True → last 20 points are novel anomalies
    - Else → anomalies randomly across whole series (outlier detection)
    """
    n_points = 200
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_points)][::-1]

    # Base normal signal
    values = np.sin(np.linspace(0, 8 * np.pi, n_points)) + np.random.normal(0, 0.2, n_points)

    df = pd.DataFrame({
        "sn": ["TEST_SN_1"] * n_points,
        "hour": timestamps,
        "avg_5gsnr": values
    })

    # Inject anomalies
    if for_novelty:
        df.loc[n_points - 20:n_points - 10, "avg_5gsnr"] += 4     # high spikes
        df.loc[n_points - 10:, "avg_5gsnr"] -= 4                  # deep drops
    else:
        anomaly_indices = np.random.choice(n_points, size=10, replace=False)
        df.loc[anomaly_indices, "avg_5gsnr"] += np.random.choice([3, -3], size=10)

    return df


# ============================================================
# Test Novelty Detection
# ============================================================
def test_novelty_detection():
    df = generate_synthetic_data(for_novelty=True)

    with mlflow.start_run(run_name="KDE_Novelty_Test"):
        params = {
            "bandwidth": 0.5,
            "filter_percentile": 95,
            "threshold_percentile": 95,
            "anomaly_direction": "both",
        }
        mlflow.log_params(params)

        detector = FeaturewiseKDENoveltyDetector(
            df=df,
            feature_col="avg_5gsnr",
            time_col="hour",
            train_idx=slice(0, -20),     # train on earlier data
            new_idx=slice(-20, None),    # detect anomaly in last 20 points
            bandwidth=params["bandwidth"],
            filter_percentile=params["filter_percentile"],
            threshold_percentile=params["threshold_percentile"],
            anomaly_direction=params["anomaly_direction"],
        )

        result = detector.fit()
        # Get figures
        fig1 = detector.plot_line()
        fig2 = detector.plot_kde()

        # Save to MLflow
        mlflow.log_figure(fig1, "kde_timeseries_plot.png")
        mlflow.log_figure(fig2, "kde_distribution_plot.png")

        mlflow.log_metric("n_train_samples", len(df.iloc[: -20]))
        mlflow.log_metric("n_test_samples", len(df.iloc[-20:]))
        mlflow.log_metric("n_anomalies_detected", len(result))

        print("\n📌 Novelty Detection Results:")
        print(result)

    return result


# ============================================================
# Test Outlier Detection (Full-Series)
# ============================================================
def test_outlier_detection():
    df = generate_synthetic_data(for_novelty=False)

    with mlflow.start_run(run_name="KDE_Outlier_Detection_Test"):
        params = {
            "bandwidth": 0.5,
            "filter_percentile": 95,
            "threshold_percentile": 95,
            "anomaly_direction": "both",
        }
        mlflow.log_params(params)

        detector = FeaturewiseKDENoveltyDetector(
            df=df,
            feature_col="avg_5gsnr",
            time_col="hour",
            train_idx="all",     # full training data
            new_idx="all",       # test all (outlier detection)
            bandwidth=params["bandwidth"],
            filter_percentile=params["filter_percentile"],
            threshold_percentile=params["threshold_percentile"],
            anomaly_direction=params["anomaly_direction"],
        )

        result = detector.fit()
        # Get figures
        fig1 = detector.plot_line()
        fig2 = detector.plot_kde()

        # Save to MLflow
        mlflow.log_figure(fig1, "kde_timeseries_plot.png")
        mlflow.log_figure(fig2, "kde_distribution_plot.png")

        mlflow.log_metric("n_total_samples", len(df))
        mlflow.log_metric("n_anomalies_detected", len(result))

        print("\n📌 Outlier Detection Results:")
        print(result)

    return result


# ============================================================
# Entry Point for Local Testing
# ============================================================
if __name__ == "__main__":
    print("\n========== Running Novelty Detection Test ==========")
    novelty_results = test_novelty_detection()

    print("\n========== Running Outlier Detection Test ==========")
    outlier_results = test_outlier_detection()

    print("\n🎯 Testing Completed.")
