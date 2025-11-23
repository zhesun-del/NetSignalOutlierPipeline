# NetSignalOutlierPipeline
A scalable, Spark-powered anomaly detection platform for analyzing WiFi/5G network telemetry (RSRP, SNR, throughput, CQI, packet bytes). It transforms raw station logs into model-ready time series, applies statistical and deep learning-based anomaly detection models, tracks experiments with MLflow, and supports deployment for real-time and batch monitoring.


```shell
./run_spark.sh tests/test_pipeline_integration.py
```