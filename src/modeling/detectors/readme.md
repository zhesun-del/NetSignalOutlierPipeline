
# 📡 Wi-Fi Score Analysis

Real-time performance monitoring, anomaly detection, and novelty detection for 5G Home Wi-Fi customer experience.

---

## 📊 Outlier Detection

Detects abnormal points within the same data distribution, without assuming the training data is clean.

| Outlier Detection - kernel density estimation |
|-----------------------------|
| <img width="800" height="507" alt="Screenshot 2025-11-29 at 3 19 36 PM" src="https://github.com/user-attachments/assets/4c7eb98e-5f8f-4e14-a49e-7c2a10a96abe" /> |

| Outlier Detection - arima |
|-----------------------------|
| <img width="800" height="568" alt="Screenshot 2025-11-29 at 3 17 23 PM" src="https://github.com/user-attachments/assets/548667d3-295c-4f1b-a721-efddb873ee59" /> |

| Outlier Detection - autoencoder |
|--------------------------------|
| <img width="800" height="610" alt="Screenshot 2025-11-29 at 3 21 31 PM" src="https://github.com/user-attachments/assets/5a6f1420-db44-4d85-a4cb-68f949f6a5a1" /> |

---

## 🧠 Novelty Detection

Assumes the training data is normal only, learns its patterns, and detects new, previously unseen anomalies in future data.

| Novelty Detection - kernel density estimation |
|----------------------------------------|
| <img width="700" height="517" alt="Screenshot 2025-11-29 at 3 23 35 PM" src="https://github.com/user-attachments/assets/bcebdb22-e7e6-4d46-bedc-3ac9467f446b" /> |

| Novelty Detection - autoencoder |
|--------------------------------------|
| <img width="700" height="502" alt="Screenshot 2025-11-29 at 3 24 33 PM" src="https://github.com/user-attachments/assets/189172df-4715-4780-92dd-d82e9e3c9cb2" /> |

---

for details about each detector, check 
- https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/README.md
- for difference between outlier detection and novelty detection
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/README.md#4-novelty-detection-vs-outlier-detection
- for different detector
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/README.md#a-extreme-value-no-time-dependence
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/README.md#b-time-series-with-time-dependence
  - https://github.com/GeneSUN/Anomaly_Detection_toolkit/blob/main/README.md#c-unusual-shape-subsequence-anomalies
