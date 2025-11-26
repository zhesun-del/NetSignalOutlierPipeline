# State of Art of real-time Anomaly Detection

| Dimension | Axis | Options |
|-----------|------|---------|
| **Granularity (Who is the model for?)** | Global → Segment → Personalized → Per-Customer |
| **Lifecycle (How long is the model valid?)** | Static → Periodic → Online → Disposable |

## 🌐 Dimension 1: *Granularity of Modeling*

The first dimension is **granularity** 
- ranging from global models to individual (per-customer) models.
- In between these two extremes lies the **segment-based approach**,
  - where customers are grouped into cohorts or clusters with similar behavior/ metadata (plan type, geography, device type, usage intensity, etc.).
  - This creates a semi-global or multi-customer model that balances scalability and personalization.

```
                     ┌───────────────────────────────┐
                     │       Global Model            │
                     │  Trained on ALL customers     │
                     │  {A1,A2,A3,B1,B2,B3,C1,C2,C3} │
                     └───────────────────────────────┘
                                   ↓
       ┌───────────────────────────────────────────────────────────────┐
       │         Segment-Based / Cohort Models (Semi-Global)          │
       │                                                               │
       │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
       │   │  Model A     │   │  Model B     │   │  Model C     │      │
       │   │ {A1,A2,A3}   │   │ {B1,B2,B3}   │   │ {C1,C2,C3}   │      │
       │   └──────────────┘   └──────────────┘   └──────────────┘      │
       └───────────────────────────────────────────────────────────────┘
                                   ↓
       ┌───────────────────────────────────────────────────────────────┐
       │                 Individual / Per-Customer Models              │
       │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
       │   │ A1  │ │ A2  │ │ A3  │ │ B1  │ │ B2  │ │ B3  │ │ C1  │ ... │
       │   │M_A1 │ │M_A2 │ │M_A3 │ │M_B1 │ │M_B2 │ │M_B3 │ │M_C1 │     │
       │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘     │
       └───────────────────────────────────────────────────────────────┘



Global ──────────────┬───────────── Segment-Based ──────────────┬───────────── Individual
(model for all)      │       (cluster/cohort models)            │    (per-customer model)
                     │                                          │
         Semi-Global / Multi-Customer                    Highly Personalized

```


### 1. Global-but-personalized models (multi-entity TS, entity embeddings)

This is basically the **DeepAR / TFT style** approach:

- Train **one big sequence model** (RNN/Transformer/Temporal CNN) on *all* customers.
- Feed **customer ID (or customer embedding + metadata)** into the model as extra features.
- The model learns:
  - **Global regularities** (diurnal / weekly patterns, holidays, network-wide events).
  - **Per-customer idiosyncrasies** via embeddings.

**Pros**
- You automatically get personalization without storing a separate model per customer.
- Works well when you have tons of related time series.

**Cons**
- Harder to interpret than simple per-customer rules/stat models.
- Needs careful handling of concept drift and new customers.
- Deployment complexity (GPU/serving, latency).



### 2. Two-Stage Detection: cheap global + expensive local

To cope with scale, use **two stages**:

1. **Stage 1: Very cheap filter** (global thresholds, global model, or simple per-customer stats).
   - Goal: filter 99% of “obviously normal” events.

2. **Stage 2: Heavy model only for suspicious candidates**
   - Could be:
     - Per-customer on-the-fly model,
     - A global-but-personalized deep model,
     - A full embedding + nearest neighbor search.

more general, this could be Hierarchical / multi-level detectors (customer → cell → region → global)


**Pros**
- Much cheaper than running heavy models everywhere.

**Cons**
- Risk of missing anomalies if Stage 1 is too aggressive.
- Pipeline complexity (two types of models, two latencies).



### 3. Meta-learning “fast per-customer adaptation”

Use **meta-learning**:

1. Train a meta-model across all customers so that:
2. Given a small amount of recent data from a new customer,
3. You can quickly adapt to this customer with 1–2 gradient steps (MAML-style) or fine-tuning of a small head.

Workflow:

1. Offline: meta-train across many customers’ histories.
2. Online: when you get a new/changed customer:
   - Start from the meta-initialization,
   - Do a tiny bit of adaptation on their most recent history,
   - Use the adapted model for anomaly scoring.

this is similar to llm few-shot methodology. you train a large model use customer information, and then fine-tune adapt to personalized information.

**Pros**
- Gets close to “per-customer custom models” without full retraining each time.
- Handles dynamic customers better than a static global model.

**Cons**
- Research-y; serving & training infra is not trivial.
- Needs enough historical customers to meta-train.


---

## ⏳ Dimension 2: *Lifecycle of Training & Usage*
This dimension concerns **how long a model remains valid before retraining**—which is especially important in real-time detection where concept drift is high.

| Lifecycle Strategy | Description | Suitable For |
|--------------------|-------------|--------------|
| **Static (Train Once, Use Long-Term)** | Train once, reuse model for weeks/months. | Stable domains (like house pricing) |
| **Periodic Retraining (Batch)** | Retrain daily/weekly using new data. | Medium-drift environments |
| **Incremental / Online Learning** | Model updates continuously as new data arrives. | Streaming, evolving data |
| **Disposable (Train-Use-Discard)** | Model trained on historical window, used once to detect anomalies, then discarded. | Highly dynamic environments (novelty, drift) |


In real-time anomaly detection, data is **highly time-sensitive**, and historical patterns may quickly become irrelevant. Therefore:

> A model trained using Day 1–14 may **not** be reliable to detect anomalies in Day 20.  
> Because user behavior, network conditions, or external dynamics **change too rapidly**.

Thus, **Disposable or Online models** are often preferred for novelty detection.

### Conclusion
This approach builds a **temporary, disposable model** for each customer using their most recent historical data — **only when needed**.  
It does **not maintain or store** long-term models. Instead, it **re-trains every time** you want to detect anomalies.

```
Historical Window (Training Data)                      New Point (Test)          Model Usage
─────────────────────────────────────────────────     ──────────────────    ────────────────────────────
1, 2, 3, ..., 98, 99, 100      ──▶ Train Model M₁ ──▶   101   → anomaly?   → M₁ DISCARDED after test

2, 3, 4, ..., 99, 100, 101     ──▶ Train Model M₂ ──▶   102   → anomaly?   → M₂ DISCARDED after test

3, 4, 5, ..., 100, 101, 102    ──▶ Train Model M₃ ──▶   103   → anomaly?   → M₃ DISCARDED after test

4, 5, 6, ..., 101, 102, 103    ──▶ Train Model M₄ ──▶   104   → anomaly?   → M₄ DISCARDED after test

... and so on — continuous sliding window detection

```

---


## 2. How tech companies handle massive real-time anomaly detection

Here are some concrete systems you can study and steal ideas from:

- **LinkedIn – ThirdEye**
  - ThirdEye is LinkedIn’s business-wide monitoring platform for detecting anomalies in KPI time series across many products and dimensions (country, segment, experiment, etc.).  
  - It supports multiple algorithms, automatic root cause analysis over high-cardinality dimensions, and smart alerting tuned to business sensitivity.
    
- **Uber – M3 + uVitals + alerting ecosystem**
  - **M3**: Uber’s open-source, large-scale metrics platform used as remote storage for Prometheus; designed explicitly for multi-tenant, high-cardinality metrics (thousands of services, billions of time series).  
  - **uVitals**: newer anomaly detection and alerting system specialized for multi-dimensional time-series data, working in an unsupervised fashion for service health. 
  - **Observability at scale**: Uber describes how they built metrics & alerting pipelines (uMonitor, Neris) on top of this stack.

- **Twitter / X – AnomalyDetection & S-H-ESD**
  - Twitter’s engineering blog details how they built practical and robust anomaly detection in time series using Seasonal Hybrid ESD, including treatment of seasonal patterns and long histories. 
  - Many open-source reimplementations exist; these are widely used as baselines for metric anomaly detection.

- **Grafana Labs – Prometheus + Mimir anomaly detection**
  - Grafana Cloud runs multi-tenant metrics storage (Mimir) and has shared how they implement anomaly detection rules over Prometheus-style metrics at scale. This is very close to your “hundreds of thousands customers” case if you treat each customer × metric pair as a series. 

- **Amazon – DeepAR / Amazon Forecast**
  - DeepAR is used internally (and via Amazon Forecast) for large-scale forecasting across many SKUs / entities; anomaly detection is often implemented as “forecast + residual thresholding” on top of these models. 

- **Commercial / infra products**
  - **StarTree ThirdEye**: productized anomaly detection and root-cause analysis for OLAP metrics (built on Apache Pinot), used by companies like Confluent and Walmart. 
  - **Enterprise TS DBs & observability tools** (VictoriaMetrics Enterprise, Cortex, etc.) often ship anomaly detection and multi-tenant statistics specifically for large numbers of metric streams. 

---


## 3. Resources (papers, docs, libraries) tied to the above ideas

Here are some concrete things to read / look at, mapped loosely to the options:

- **Global-but-personalized time-series models (Option 5, 12)**
  - DeepAR paper: probabilistic forecasting with a single RNN trained over many related time series; the ideas map directly to global anomaly thresholds on forecast residuals. 
  - Amazon Forecast DeepAR+ docs show how this style of model is productized for multi-entity forecasting. 

- **Robust seasonal decomposition + anomaly detection (Options 1/4/7/9)**
  - Twitter’s Seasonal Hybrid ESD (S-H-ESD) for time-series anomalies; used in their internal AnomalyDetection tool and widely replicated.
  - “Enhanced Seasonal-Hybrid ESD (SH-ESD+)” extends this idea for more robust detection, especially for long seasonal series.

- **Multi-dimensional / multi-tenant anomaly platforms (Options 4/8/10/11)**
  - LinkedIn’s ThirdEye: end-to-end anomaly detection, smart alerts, and root-cause for business metrics; their blogs discuss multi-dimensional metrics, grouping, and high cardinality. 
  - StarTree ThirdEye (commercialization) documents how companies like Walmart and Confluent use it for business monitoring at scale.

- **Streaming / metrics-focused anomaly detection (Options 7/8/10)**
  - Grafana’s “How to use Prometheus to efficiently detect anomalies at scale” describes large-scale, explainable anomaly detection on multi-tenant metrics (using Mimir). 
  - Uber’s anomaly detection platform blog (“Implementing Model-Agnosticism in Uber’s Real-Time Anomaly Detection”) describes a model-agnostic pipeline that can host multiple algorithms behind one alerting system. 
- **Federated / multi-tenant research (Option 11)**
  - Recent work on “Federated Anomaly Detection for Multi-Tenant Cloud Environments” proposes federated detection for dynamic, high-cardinality tenants. 
These are good starting points to see how your conceptual options show up in real code / systems.

---
