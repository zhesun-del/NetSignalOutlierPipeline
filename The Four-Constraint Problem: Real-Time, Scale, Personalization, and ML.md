# The Four-Constraint Problem: Real-Time, Scale, Personalization, and ML


## 1) Project context (what we’re trying to do)

- Population:
  - ~2 million Wi-Fi home customers (each “customer” = one home/router entity)
  - Each router has ~20 important features
  - Total “customer-feature” series = 2,000,000 × 20 = 40,000,000

- Data cadence:
  - Every customer generates a log every 5 minutes
  - Each log contains all 20 features at that timestamp

- Desired output:
  - Every 5 minutes, the system produces predictions/anomaly signals for:
    - every customer AND every feature
  - In other words: **“40 million customer-feature predictions every 5 minutes”**

## 2) Manager’s proposed modeling requirement: “customized”

- The proposal is NOT a standard global model such as:
  - “Train one model on feature signal-strength across all 2M customers”
  - That approach assumes a shared pattern and generalizes across customers

- Instead, “customized” means:
  - Customer 1’s signal-strength behavior is treated as fundamentally different from Customer 2’s signal-strength behavior
  - The model must adapt per-customer-per-feature
  - Conceptually, it implies something like:
    - a separate model
    - or separate parameters / separate decision boundary
    - or separate calibration/thresholds
    for EACH (customer, feature) pair

- That is the “customized global model” challenge:
  - We want ML/DL, but the behavior is individualized at the finest **granularity**:
    (customer × feature), not population-level.


## 3) The four major challenges (clearly stated)

**Challenge #1 — Real-time SLA**
- Must generate results every 5 minutes (end-to-end).
- This includes ingestion → compute → write/serve.

**Challenge #2 — Large population scale**
- 40 million customer-feature entities to score repeatedly.
- With 5-minute cadence, the scoring volume is enormous.

**Challenge #3 — Per-customer-per-feature customization**
- Not “one model per feature across customers.”
- Not “one global model for all customers.”
- Requirement is effectively “personalized” modeling for each customer-feature.

**Challenge #4 — Machine learning computational cost**
- ML/DL inference and/or training is more expensive than rule-based calculations
  (e.g., percentiles, thresholds, z-scores).
- Complexity increases further when combined with customization and real-time SLA.


## 4) Reality check using the existing hourly-score project

Current system:

- Hourly score pipeline exists:
  - Rule-based(not ml or customization), 5-7 features
  - Just calculate data to HDFS already takes ~15 minutes
  - Updating/serving via Druid takes additional time

- 5g-home-score:
  - model/price-plan/percentitle(simple customization), 15+ features
  - Just calculate data to HDFS already takes ~40 minutes(2173 s)


- Implication:
  - If “hourly” already struggles with latency components > 5 minutes,
    **then “5-minute full cycle” is not feasible under current architecture/resources.**
  - The bottleneck is not just **modeling** — it’s **ingestion** + **storage** + **serving latency**.


## 5) Feasibility statement (what is possible vs. not possible)

**“5-minute full cycle” is not feasible under current architecture/resources.**

Given current resources, we can realistically satisfy only ~2 of the 3 constraints at once.

A) ML/DL + Large population (40M customer-features)
- Possible direction:
  - Train a global/shared model (or a small set of segment models)
  - Run distributed inference at scale
- Not satisfied:
  - True per-customer-per-feature customization is lost

The brutal conclusion，Under real Spark + MLflow + deep learning inference:

| Model                 | Time for 40M series |
| --------------------- | ------------------- |
| Small MLP autoencoder | ~15–30 minutes      |
| LSTM autoencoder      | ~1–2 hours          |
| Larger models         | multiple hours      |


B) Customization + Large population (40M)
- Possible direction:
  - Simple individualized stats/rules per customer-feature(rolling mean/std, quantiles, EWMA, dynamic thresholds)
  - IQR Udf for 10 features (～10 minutes, 528 seconds)  
- Not satisfied:
  - It’s not really “machine learning” in the deep/model sense
  - More like scalable personalized analytics

C) Customization + ML/DL
- Possible direction:
  - Per-customer models or meta-learning style personalization
- Not satisfied:
  - Scale collapses at 40M customer-features (training, storage, monitoring, deployment)
  - Operationally not feasible with current compute/time constraints

D) Real-time 5-minute SLA + any heavy combination
- With current pipeline latency (HDFS + Druid update time),
  end-to-end 5-minute SLA is not achievable regardless of model choice,
  unless architecture changes (streaming/low-latency infra) are introduced.


## 6) Bottom-line conclusion (the decision framing)

- The ask combines four hard constraints simultaneously:
  (5-minute real-time) + (40M scale) + (per customer-feature customization) + (ML/DL).
- Under current architecture/resources, this full combination is not feasible.
- What is feasible is choosing a subset:
  - either scalable ML at population level (shared pattern),
  - or scalable customization using lightweight rules/stats,
  - or ML customization but only for a smaller subset / sampled population / top customers,
  - and real-time requires infra changes beyond the modeling layer.
