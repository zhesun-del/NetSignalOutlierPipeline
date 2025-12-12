# NetSignalOutlierPipeline



## 🔹 Overview

This pipeline performs real-time anomaly detection on network signal data using: 

- Custom preprocessing(missing handling, incremental transform, normalization)
- Spark distributed computing
- KDE + EWMA + ARIMA ensemble detection
- MLflow experiment tracking


<img width="1200" height="1505" alt="Untitled" src="https://github.com/user-attachments/assets/8ec85b4f-2d5e-4f71-b23a-280ddce40a6a" />

## 📚 Table of Contents

- [🚀 Pipeline Flow](#-pipeline-flow)
  - [1️⃣ Data Ingestion (HDFS)](#1️⃣-data-ingestion-hdfs)
  - [2️⃣ Preprocessing](#2️⃣-preprocessing)
  - [3️⃣ Wide → Long Format](#3️⃣-wide--long-format)
  - [4️⃣ Distributed Detection (Spark)](#4️⃣-distributed-detection-spark)
  - [5️⃣ Model Ensemble](#5️⃣-model-ensemble)
  - [6️⃣ MLflow Tracking](#6️⃣-mlflow-tracking)
  - [📤 Output Schema](#output-schema)

------------------------------------------------------------------------

## 🚀 Pipeline Flow

Step 1,2,3 are data process, more details discussed in 
- https://github.com/GeneSUN/NetSignalOutlierPipeline/tree/main/src/Preprocessing

### 1️⃣ Data Ingestion (HDFS)

-   Reads hourly partitions from HDFS using rolling window.
-   Supports dynamic path generation: `/YYYY-MM-DD/hr=HH`.

### 2️⃣ Preprocessing

| Step                     | Description                         |
|--------------------------|-------------------------------------|
| convert_string_numerical | Convert features to numeric         |
| forward_fill             | Fill zero by SN/time      |
| HourlyIncrementProcessor | Apply increment, log, and fill missing     |
| orderBy                  | Ensure time-order per SN            |


------------------------------------------------------------------------

### 3️⃣ Wide → Long Format

Transforms:

    sn | time | RSRP | RSRQ | TotalBytes...

→

    sn | time | feature | value

------------------------------------------------------------------------

Step 4,5,6 are modeling, more details discussed in: 
- https://github.com/GeneSUN/NetSignalOutlierPipeline/tree/main/src/modeling/distributed_detection_runtime
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/tree/main
  
### 4️⃣ Distributed Detection (Spark)

Spark `applyInPandas` performs feature-wise group-level anomaly
detection:

```python
spark = SparkSession.builder.appName("NetSignalDev").getOrCreate()
sc = spark.sparkContext
sc.addPyFile("hdfs:///libs/net_signal_pipeline.zip")
```

``` python
df_long.groupBy("sn", "feature").applyInPandas(...)
```

------------------------------------------------------------------------

### 5️⃣ Model Ensemble

| Model     | Purpose                                      |
|-----------|----------------------------------------------|
| KDE       | Density-based abnormal probability detection |
| EWMA      | Detects sudden behavioral level shifts       |
| ARIMA     | Captures temporal trend & seasonality-based anomalies |
| Ensemble  | Flags anomaly if any model detects abnormality |

------------------------------------------------------------------------

### 6️⃣ MLflow Tracking

| Category          | Details                                                                        |
| ----------------- | -------------------------------------------------------------------------------|
| **Parameters**    | Window size, percentiles, scaling mode, detector configs, direction (high/low) |
| **Metrics**       | `n_points`, `anomalies_detected`, true/false counts, processing time           |
| **Run Hierarchy** | Nested tracking: `Experiment → SN → Feature`                                   |
| **Error Logging** | Exceptions captured using `mlflow.log_param("error", ...)`                     |
| **Artifacts**     | 📎 Time-series anomaly plots |

<img width="753" height="654" alt="image" src="https://github.com/user-attachments/assets/c902dedf-6f39-4236-88a7-ed94960ec8fc" />

<img width="1423" height="827" alt="image" src="https://github.com/user-attachments/assets/5c86e4c8-9cef-43bc-ac7a-ff7526db881a" />


------------------------------------------------------------------------

### Output Schema

    sn | time | feature | value | is_outlier_kde | is_outlier_ewma | is_outlier_arima

------------------------------------------------------------------------
