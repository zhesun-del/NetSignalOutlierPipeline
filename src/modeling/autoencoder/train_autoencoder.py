from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,percentile_approx,explode

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# train_autoencoder.py

import mlflow
import mlflow.pyfunc
import pandas as pd
import  sys
sys.path.append('/usr/apps/vmas/scripts/ZS/NetSignalOutlierPipeline/src/modeling/autoencoder') 

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
    ae = MultiTimeSeriesAutoencoder(
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
            registered_model_name="Autoencoder_Anomaly_Detection"  # 👈 THIS IS NEW
        )

        print("Model, metrics, and plots logged to MLflow successfully.")

    return ae



def read_data():

    df_slice = spark.read.option("recursiveFileLookup", "true").parquet("/user/ZheS/owl_anomaly/autoencoder/data/")

    pdf = df_slice.filter( col("sn")=="ACL35000028" ).filter(col("feature")=="4GRSRP").toPandas()



    pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
    pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
    pdf = pdf.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    return pdf



LOOKBACK_DAYS = 30
WINDOW_SIZE = 24
OVERLAP = 0.5
THRESHOLD_PERCENTILE = 99
SCALER = "standard"   # or "minmax" or None


if __name__ == "__main__":

    spark = (
        SparkSession.builder
        .appName("Autoencoder_Training")
        .config("spark.ui.port", "24045")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")    

    mlflow.set_tracking_uri("http://njbbvmaspd11:5001")
    #mlflow.set_tracking_uri("file:/usr/apps/vmas/scripts/ZS/mlflow")
    mlflow.set_experiment("Autoencoder_Anomaly_Detection")
    pdf=read_data()
    train_and_log(
        df=pdf[["sn","time","value","slice_id"]],
        time_col="time",
        feature="value",
        slice_col="slice_id",
        model_params={"epochs": 10},
    )