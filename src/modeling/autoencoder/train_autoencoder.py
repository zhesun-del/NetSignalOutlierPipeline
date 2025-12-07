# train_autoencoder.py

import mlflow
import mlflow.pyfunc
import pandas as pd
from TimeSeriesAutoencoderTrainer import TimeSeriesAutoencoderTrainer
from inference_wrapper import AutoencoderWrapper


def train_and_log(
    df: pd.DataFrame,
    time_col: str,
    feature: str,
    slice_col: str = "slice_id",
    model_params: dict = None,
    scaler="standard",
    threshold_percentile=99,
):
    ae = TimeSeriesAutoencoderTrainer(
        df=df,
        time_col=time_col,
        feature=feature,
        slice_col=slice_col,
        model_params=model_params,
        scaler=scaler,
        threshold_percentile=threshold_percentile,
    )

    ae.prepare()

    with mlflow.start_run():
        # Log params
        mlflow.log_params(ae.get_params())

        # Train
        ae.fit()

        # Metrics
        mlflow.log_metric("threshold", float(ae.threshold_scores))
        mlflow.log_metric("mean_score", float(ae.anomaly_scores.mean()))
        mlflow.log_metric("std_score", float(ae.anomaly_scores.std()))

        # ----------------------------------------------------
        # Log Plots
        # ----------------------------------------------------
        fig1 = ae.plot_anomaly_score_distribution()
        mlflow.log_figure(fig1, "plots/anomaly_score_distribution.png")

        fig2 = ae.plot_time_series_by_category("normal")
        mlflow.log_figure(fig2, "plots/time_series_normal.png")

        fig3 = ae.plot_mean_and_spread()
        mlflow.log_figure(fig3, "plots/mean_and_spread.png")

        # Save model
        ae.save_model("autoencoder.pkl")
        mlflow.log_artifact("autoencoder.pkl")

        # Pyfunc model
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=AutoencoderWrapper(),
            artifacts={"autoencoder": "autoencoder.pkl"},
        )

        print("Model, metrics, and plots logged to MLflow successfully.")

    return ae


if __name__ == "__main__":
    df = pd.read_csv("example_timeseries.csv")
    train_and_log(
        df=df,
        time_col="timestamp",
        feature="value",
        slice_col="slice_id",
        model_params={"epochs": 10},
    )
