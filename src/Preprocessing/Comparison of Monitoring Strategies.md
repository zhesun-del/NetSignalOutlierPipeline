# Comparison of Monitoring Strategies: 5-Minute Realtime vs Hourly Aggregation

## 1. Context

This analysis focuses on **5G Home Fixed Wireless Access (FWA)** network performance.  
Each feature, including `lte_capacity`, is recorded every **5 minutes**, yielding **12 samples per hour** per device.

Two monitoring strategies are under consideration:

1. **Realtime 5-minute monitoring** — detects short-term fluctuations and enables rapid anomaly response, aligning with the vision for near real-time insights.  
2. **Hourly aggregation** — consolidates twelve 5-minute readings into a single hourly record, offering greater stability, clearer trends, and more scalable processing.

---

### (a) Realtime 5-Minute Data
<img width="2048" height="826" alt="image" src="https://github.com/user-attachments/assets/8f0f6deb-36c4-4727-bf01-cf34a565e6ca" />
Pros:
- real-time monitoring promises immediacy, provides high granularity

Cons:
- The raw five-minute data shows significant volatility, creating **noise** that can easily trigger **false positives** in novelty detection models.  
-  **Computationally expensive** to process at such frequency — particularly across tens of thousands of customers.

---

### (b) Hourly Aggregation
<img width="2024" height="832" alt="image" src="https://github.com/user-attachments/assets/30dda8ef-c9f5-40a2-9ae5-2c537ac61162" />

Aggregating into hourly averages (`avg_lte_capacity`, `avg_lte_band`, `avg_4gsnr`) smooths the trends considerably.  
- provides a balanced view that maintains operational sensitivity without overwhelming system resources.

---

## 2. Exploring Acceleration

Potential strategies to accelerate real-time anomaly detection:

- [**Anomaly_Detection_Toolkit** – repository overview](https://github.com/GeneSUN/Anomaly_Detection_toolkit/tree/main)  
- [**Section 5: Multi-Models Distributed Computing**](https://github.com/GeneSUN/Anomaly_Detection_toolkit/tree/main?tab=readme-ov-file#5-multi-models-distributed-computing)

These approaches include:
- Distributed model parallelization  
- Lightweight ensemble scoring  

---

## 3. Performance Collection

To assess real-time scalability, Spark runtime benchmarks were collected.

### ⚙️ Spark Job Configuration

| Parameter | Value |
|------------|--------|
| Customer number | 110 k |
| Executor cores | 700 |
| `spark.sql.shuffle.partitions` | 2800 |
| **Total Runtime** | **6.0 min** |

### 🧾 Spark Job Summary

| Job ID | Description | Submitted | Duration | Stages (Succeeded/Total) | Tasks (Succeeded/Total) |
|--------:|--------------|------------|-----------|----------------------------|----------------------------|
| 0 | Listing leaf files and directories for 2400 paths | 9/8/25 19:16 | 5 s | 45658 | 2400 / 2400 |
| — | `hdfs://njbbepapa1.nss.vzwnet.com:9000/user/kovvuve/owl_history_v3/date=2025-09-08/hr=19/...csv.gz` | — | — | — | — |
| 1 | csv at `NativeMethodAccessorImpl.java:0` | 9/8/25 19:16 | 0.9 s | 45658 | 1 / 1 |
| 2 | first at `NoveltyRealtimePipeline.py:634` | 9/8/25 19:16 | 45 s | 45658 | 2400 / 2400 |
| 3 | first at `NoveltyRealtimePipeline.py:634` | 9/8/25 19:17 | 17 s | 1 / 1 (1 skipped) | 2800 / 2800 (2400 skipped) |
| 4 | first at `NoveltyRealtimePipeline.py:634` | 9/8/25 19:17 | 0.1 s | 1 / 1 (2 skipped) | 1 / 1 (5200 skipped) |
| 5 | parquet at `NativeMethodAccessorImpl.java:0` | 9/8/25 19:17 | 1.1 min | 45719 | 8000 / 8000 |
| 6 | parquet at `NativeMethodAccessorImpl.java:0` | 9/8/25 19:18 | 3.3 min | 4 / 4 (2 skipped) | 11200 / 11200 (5200 skipped) |

---

### Model Processing Time by Device Type

| ModelName | Count | Time |
|------------|-------:|------:|
| **ALL model** | 3,194,212 | **60.00 min** |
| XCI55AX | 985,718 | 22.95 min |
| WNC-CR200A | 938,145 | 23.88 min |
| ASK-NCQ1338FA | 394,185 | 11.47 min |
| ASK-NCM1100 | 283,946 | 9.62 min |
| ASK-NCM1100E | 165,443 | 6.74 min |
| FSNO21VA | 132,673 | 6.27 min |
| ASK-NCQ1338 | 104,103 | 5.74 min |
| FWF100V5L | 70,202 | 4.58 min |
| XC46BE | 63,795 | 4.32 min |
| ASK-NCQ1338E | 30,705 | 3.39 min |
| FWA55V5L | 25,297 | 3.32 min |

Even with heavy optimization (`700 cores`, `2800 shuffle partitions`), processing all 3.2 million customer models every 5 minutes remains challenging.  
The runtime and data volume make full real-time detection difficult to sustain continuously.


---

## 6. Summary

| Approach | Description | Pros | Cons |
|-----------|--------------|------|------|
| **5-minute real-time monitoring** | Detects anomalies quickly from raw data | High resolution; fast reaction time | Noisy, unstable, and expensive to compute |
| **Hourly aggregation** | Aggregates 12 five-minute windows | Smooth, interpretable, and efficient | Slight delay in detection; less granular |


📈 **Conclusion:**  
While real-time detection offers immediacy, its cost and instability outweigh the benefits at current scale.  
**Hourly aggregation** provides a **balanced compromise** — smoother signals, lower cost, and easier integration into distributed anomaly pipelines.
