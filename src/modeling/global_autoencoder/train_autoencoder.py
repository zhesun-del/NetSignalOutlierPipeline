from pyspark.sql.functions import sum, lag, col, split, concat_ws, lit ,udf,count, max,lit,avg, when,concat_ws,percentile_approx,explode

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import warnings
warnings.filterwarnings("ignore")

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

        # Pyfunc model, register a wrapped inference model, NOT the trainable PyTorch model
        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=AutoencoderWrapper(),
            artifacts={"autoencoder": "autoencoder.pkl"},
            registered_model_name="Autoencoder_Anomaly_Detection"  # 👈 THIS IS NEW

        )
        '''
        pyfunc_model/
            MLmodel
            python_model.pkl   ← wrapper
            autoencoder.pkl    ← your actual PyTorch model weights (inside artifacts)
        python_model.pkl contains your AutoencoderWrapper, which exposes:
            .predict()
            .load_context()
            not .fit(), not .train(), not .optimizer, not .forward() raw access
        '''

        print("Model, metrics, and plots logged to MLflow successfully.")

    return ae

def train_and_log_incremental(
    df_new: pd.DataFrame,
    time_col: str,
    feature: str,
    slice_col: str,
    base_model_uri: str,    # ← this tells us which MLflow model to resume
    model_params=None,
    scaler="standard",
    threshold_percentile=99,
):
    # 1) Download last model
    local_path = mlflow.pyfunc.load_model(base_model_uri).artifact_path
    checkpoint_path = local_path + "/autoencoder.pkl"

    # 2) Load + initialize trainer
    ae = TimeSeriesAutoencoderTrainer(
        df=df_new,
        time_col=time_col,
        feature=feature,
        slice_col=slice_col,
        model_params=model_params,
        scaler=scaler,
        threshold_percentile=threshold_percentile,
    )
    ae.prepare()
    ae.load_checkpoint(checkpoint_path)  # ← NEW

    # 3) Start new run (creates v2, v3, v4…)
    with mlflow.start_run():
        mlflow.log_params(ae.get_params())

        # 4) Continue training
        ae.fit(epochs=5)  # add only a few more epochs, not full re-train

        # 5) Log new metrics + save updated checkpoint
        ae.save_model("autoencoder.pkl")
        mlflow.log_artifact("autoencoder.pkl")

        mlflow.pyfunc.log_model(
            artifact_path="pyfunc_model",
            python_model=AutoencoderWrapper(),
            artifacts={"autoencoder": "autoencoder.pkl"},
            registered_model_name="Autoencoder_Anomaly_Detection"
        )
    return ae

def read_data():

    df_slice = spark.read.option("recursiveFileLookup", "true").parquet("/user/ZheS/owl_anomaly/autoencoder/data/")
    pdf = df_slice.filter( col("sn")=="ACL35000028" ).filter(col("feature")=="4GRSRP").toPandas()

    pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
    pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
    pdf = pdf.dropna(subset=["time", "value"]).sort_values("time").reset_index(drop=True)

    test_df = df_slice.filter( col("feature")=="4GRSRP" )\
                        .filter( col("slice_id")=="ACN40800572_20250915175856" )\
                        .distinct()

    return pdf, test_df

def convert_single_slice(df, time_col, feature):
    pdf = df.orderBy(time_col).toPandas()
    arr = pdf.sort_values(time_col)[feature].values
    return arr.reshape(1, -1, 1)   # shape = (1, T, 1)

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
    mlflow.set_experiment("Autoencoder_Anomaly_Detection")
    pdf, test_df = read_data()
    train_and_log(
        df=pdf[["sn","time","value","slice_id"]],
        time_col="time",
        feature="value",
        slice_col="slice_id",
        model_params={"epochs": 10},
    )

    ''' Inference Example 
    model = mlflow.pyfunc.load_model("models:/Autoencoder_Anomaly_Detection/latest")

    test_tensor = convert_single_slice(test_df, "time", "value")

    result = model.predict(test_tensor)

    scores = result["scores"]
    is_outlier = result["is_outlier"]

    print("Anomaly Score:", scores[0])
    print("Is Outlier:", bool(is_outlier[0]))
    '''

    ''' Incremental Training Example
    
    ### 1. Load new week's data
    new_week_df = load_new_week_data()     # replace with your spark read

    ### 2. Get latest MLflow model
    latest_uri = "models:/Autoencoder_Anomaly_Detection/latest"

    ### 3. Incrementally train
    train_and_log_incremental(
        df_new=new_week_df,
        time_col="time",
        feature="value",
        slice_col="slice_id",
        base_model_uri=latest_uri,   # the key
        model_params={"epochs": 5},
    )

    Autoencoder_Anomaly_Detection
    v1 → full batch model
    v2 → trained with +100k new data
    v3 → trained with +100k new data
    v4 → trained with +100k new data

    '''