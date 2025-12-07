# train_autoencoder.py

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from autoencoder import MultiTimeSeriesAutoencoder
from inference_wrapper import AutoencoderWrapper


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
def train_and_log(
    df: pd.DataFrame,
    time_col: str,
    feature: str,
    slice_col: str = "slice_id",
    model_params: dict = None,
    scaler="standard",
    threshold_percentile=99,
):
    # 1. Create model
    ae = MultiTimeSeriesAutoencoder(
        df=df,
        time_col=time_col,
        feature=feature,
        slice_col=slice_col,
        model_params=model_params,
        scaler=scaler,
        threshold_percentile=threshold_percentile,
    )

    # 2. Data prep
    ae.prepare()

    # --------------------------------------------------------
    # MLflow logging
    # --------------------------------------------------------
    with mlflow.start_run():
        # log params
        mlflow.log_params(ae.get_params())

        # Train
        ae.fit()

        # log metrics
        mlflow.log_metric("threshold", float(ae.threshold_scores))
        mlflow.log_metric("mean_anomaly_score", float(ae.anomaly_scores.mean()))
        mlflow.log_metric("std_anomaly_score", float(ae.anomaly_scores.std()))

        # Save model artifact (joblib)
        ae.save_model("autoencoder.pkl")
        mlflow.log_artifact("autoencoder.pkl")

        # ----------------------------------------------------
        # Register as MLflow PyFunc model
        # ----------------------------------------------------
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=AutoencoderWrapper(),
            artifacts={"autoencoder": "autoencoder.pkl"},
        )

        print("Model logged to MLflow successfully.")

    return ae


# ------------------------------------------------------------
# Script entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("example_timeseries.csv")  # Replace with your dataset

    train_and_log(
        df=df,
        time_col="timestamp",
        feature="value",
        slice_col="slice_id",
        model_params={"epochs": 10},
        scaler="standard",
        threshold_percentile=99,
    )
