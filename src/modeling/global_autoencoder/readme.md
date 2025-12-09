# Autoencoder-Based Time Series Anomaly Detection — Training, Versioning, and Inference Pipeline

## Overview

This repository implements an anomaly detection workflow for univariate time series using deep autoencoders.

The system separates **training** and **inference** into independent modules to simplify experimentation, support **versioned model management**, and enable **safe deployment.**

The codebase is organized around three core components:

- `train_autoencoder.py` — training entry point / orchestration script  
- `TimeSeriesAutoencoderTrainer.py` — core training, evaluation, and MLflow integration  
- `inference_wrapper.py` — lightweight inference layer for deployment and API usage  


---

## Table of Contents

1. [Goal and Objectives](#goal-and-objectives)  
2. [High-Level Architecture](#high-level-architecture)  
3. [Components](#components)  
   - [train_autoencoder.py](#1-train_autoencoderpy--training-entry-point)  
   - [TimeSeriesAutoencoderTrainer.py](#2-timeseriesautoencodertrainerpy--core-training-logic)  
   - [inference_wrapper.py](#3-inference_wrapperpy--production-inference-layer)  
4. [Typical Deployment and Usage Patterns](#typical-deployment-and-usage-patterns)  
5. [Incremental Trainning](#incremental-trainning)

---

## Goal and Objectives

### Goal

Build a **anomaly detection system** that can train, evaluate, and serve autoencoder models on large-scale time series data.

### Objectives

1. Train Models
- Train autoencoder models on per-customer or per-device time series.

2. Track Experiments
- Log runs, metrics, and models with MLflow.
- Save model versions in the MLflow Model Registry.

3. Separate Training and Inference
- Keep training code separate from deployment and prediction code.
- Load a registered model and compute anomaly scores for new time series.

---

## High-Level Architecture


<img width="678" height="870" alt="image" src="https://github.com/user-attachments/assets/25369961-e88b-4e13-ad27-b23b109a93dd" />

---

## Components

```text
         ┌─────────────────────────────┐
         │      train_autoencoder.py   │
         │  (Entry point / Orchestration)
         └──────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │  TimeSeriesAutoencoderTrainer.py  │
        │  • Data prep (slice → tensor)     │
        │  • Model build & training         │
        │  • Eval & metrics                 │
        │  • MLflow logging & registry      │
        └───────────────────┬──────────────┘
                            │ registers
                            ▼
                   MLflow Model Registry
                            │ loads
                            ▼
        ┌──────────────────────────────────┐
        │        inference_wrapper.py       │
        │  • Load registry model            │
        │  • Preprocess new data            │
        │  • Compute reconstruction errors  │
        │  • Output anomaly scores          │
        └──────────────────────────────────┘
```

### 1. `train_autoencoder.py` — Training Entry Point

This script is the **main entry point** for training and registering autoencoder models. Typical responsibilities:

- Parse config / CLI arguments (paths, hyperparameters, run name, registry name, etc.).
- Load the dataset (Spark DataFrame or Pandas DataFrame).
- Apply basic filtering and splitting into train/validation sets.
- Instantiate `TimeSeriesAutoencoderTrainer` with:
  - time column
  - feature list
  - slice identifier (e.g., `slice_id`, `sn`, `mdn_5g`, etc.)
  - model and training parameters
- Trigger the full training and evaluation cycle.
- Log metrics and artifacts into MLflow.
- Optionally register the final model into the MLflow Model Registry (e.g., under a name like `Autoencoder_Anomaly_Detection`).

You can think of `train_autoencoder.py` as the **orchestration layer** for modeling and MLOps, but not for inference or serving.

---

### 2. `TimeSeriesAutoencoderTrainer.py` — Core Training Logic

This module encapsulates the **end-to-end modeling logic** and MLOps integration.

Key responsibilities:

#### Data Handling

- Convert time series data into model-ready tensors:
  - Group by `slice_id` (e.g., device/customer-level).
  - Sort by the time column.
  - Reshape into `(num_slices, time_steps, features)` for training.
- Handle univariate and multivariate cases.
- Optionally perform scaling/normalization and log transformation parameters.

#### Model Definition and Training

- Define PyTorch-based autoencoder architectures:
  - Fully connected AE
  - 1D-CNN or LSTM-based variants (optional, depending on your implementation)
- Implement training loop:
  - Forward pass → reconstruction loss (e.g., MSE)
  - Backpropagation and optimizer step
  - Validation loop with loss tracking
- Support early stopping or max-epoch-based training.

#### MLflow Integration

- Log training and validation metrics:
  - Training loss per epoch
  - Validation loss per epoch
  - Any custom metrics (e.g., reconstruction distribution stats)
- Log parameters and configs:
  - Model architecture parameters
  - Learning rate, batch size, epochs, etc.
- Log artifacts:
  - Loss curves
  - Configuration files
  - Model checkpoints if needed
- Log and **register the final model**:
  - Use `mlflow.pyfunc.log_model` or framework-specific logging.
  - Register model name and version into the MLflow Model Registry.
  - Optionally set a stage (e.g., `Staging` or `Production`).

<img width="2354" height="1506" alt="image" src="https://github.com/user-attachments/assets/ad72914f-c215-4e54-bf08-09c34f96ba48" />

**time_series_normal.png**

<img width="800" height="1000" alt="image" src="https://github.com/user-attachments/assets/431dd88c-8cb7-43d9-bbde-2811ac85b8c4" />


This module ensures that training is **reproducible, auditable, and comparable** across different runs and hyperparameter configurations.

---

### 3. `inference_wrapper.py` — Production Inference Layer

This module provides a **clean, minimal interface for anomaly scoring** from a deployed model, independent of the training pipeline.

Key responsibilities:

#### Model Loading

- Load a specific version or stage from MLflow Model Registry, for example:
  ```python
  model = mlflow.pyfunc.load_model("models:/Autoencoder_Anomaly_Detection/Production")
  ```
- Optionally support:
  - Loading by version (e.g., `.../1`)
  - Loading by stage (`Staging`, `Production`)

#### Preprocessing

- Accept new time series data in a standard format (e.g., a Pandas DataFrame with `time`, `feature`, `slice_id`).
- Apply the same preprocessing steps used in training:
  - Sorting by time
  - Reshaping to `(1, T, F)` or `(N, T, F)`
  - Applying the same scaler or normalization, if logged or serialized.

#### Scoring and Thresholding

- Run a forward pass through the model to compute **reconstruction**.
- Compute reconstruction errors and aggregate them into an anomaly score.
- Compare scores against a threshold (fixed or learned/estimated from training distributions).
- Return a standardized result, e.g.:

  ```json
  {
    "slice_id": "ACN40800572_20250915175856",
    "score": 0.034,
    "threshold": 0.017,
    "is_anomaly": true
  }
  ```

This wrapper can be used directly in:

- REST API services
- Spark batch jobs
- Streaming applications (e.g., Kafka + Spark Structured Streaming)
- Notebooks for interactive analysis


---

## Typical Deployment and Usage Patterns

The resulting models and inference wrapper can be used in multiple deployment scenarios, this project focus on Batch Scoring.

<img width="921" height="316" alt="Screenshot 2025-12-08 at 4 23 10 PM" src="https://github.com/user-attachments/assets/8833935c-36cd-4bda-8cb5-9764c74fa857" />


1. **Batch and Streaming Inference**
   - Use the wrapper inside:
     - PySpark jobs for daily or hourly batch scoring.
     - Spark Structured Streaming or Flink jobs for event-driven scoring and alerting.

This also compatible with API access:

2. **Local / On-Prem MLflow Model Server**
   - Use `mlflow models serve` or Docker to expose `/invocations` endpoint.
   - `inference_wrapper.py` can wrap the HTTP logic or be used as a library by the API server.


3. **Kubernetes + Docker**
 
4. **Cloud ML Platforms**

---


## Summary

In short, this project provides:

- A **clean training pipeline** (`train_autoencoder.py`, `TimeSeriesAutoencoderTrainer.py`)  
- A **production-ready inference wrapper** (`inference_wrapper.py`)  
- A structure that aligns with modern **MLOps practices**: experiment tracking, model registries, and safe, versioned deployment.

You can plug this into your existing Spark / MLflow / cloud ecosystem and scale from local experimentation to full production anomaly detection across thousands or millions of time series entities.



## Incremental Trainning


<img width="489" height="588" alt="Untitled" src="https://github.com/user-attachments/assets/3e691563-de11-45e6-bab6-d077e16e68f2" />
