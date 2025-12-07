# autoencoder.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder_torch import AutoEncoder


class MultiTimeSeriesAutoencoder:
    """
    Clean modeling class (NO MLflow).
    Handles:
    - data prep
    - tensor creation
    - scaling
    - training
    - thresholding
    - anomaly scoring
    - saving/loading model
    - plotting (fig returned for MLflow logging)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        feature: str,
        slice_col: str = "slice_id",
        model_params: Optional[dict] = None,
        external_model: Optional[object] = None,
        scaler: Union[str, object, None] = "None",
        threshold_percentile: float = 99,
    ):
        self.df_raw = df.copy()
        self.time_col = time_col
        self.feature = feature
        self.slice_col = slice_col
        self.model_params = model_params
        self.external_model = external_model
        self.scaler_type = scaler
        self.scaler = None
        self.model = None
        self.threshold_percentile = threshold_percentile

        self.input_tensor = None
        self.input_tensor_scaled = None
        self.anomaly_scores = None
        self.threshold_scores = None

    # --------------------------------------------------
    # Data preprocessing
    # --------------------------------------------------
    def _build_tensor_from_slices(self):
        grouped = self.df_raw.groupby(self.slice_col)
        tensors = []

        for _, group in grouped:
            series = group.sort_values(by=self.time_col)[self.feature].values
            tensors.append(series)

        tensor_3d = np.stack(tensors)[:, :, np.newaxis]
        return tensor_3d

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_type is None or self.scaler_type == "None":
            return X

        flat_X = X.reshape(-1, X.shape[-1])

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = self.scaler_type  # custom object

        flat_scaled = self.scaler.fit_transform(flat_X)
        return flat_scaled.reshape(X.shape)

    def prepare(self):
        tensor = self._build_tensor_from_slices()
        self.input_tensor = tensor
        self.input_tensor_scaled = self._apply_scaler(tensor)

    # --------------------------------------------------
    # Modeling
    # --------------------------------------------------
    def _init_model(self):
        if self.external_model:
            return self.external_model

        default_params = {
            "hidden_neurons": [
                self.input_tensor.shape[1],
                32,
                32,
                self.input_tensor.shape[1],
            ],
            "hidden_activation": "relu",
            "epochs": 20,
            "batch_norm": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.2,
        }

        if self.model_params:
            default_params.update(self.model_params)

        return AutoEncoder(**default_params)

    def fit(self, threshold_percentile=None):
        if self.input_tensor_scaled is None:
            raise ValueError("Call prepare() before fit().")

        if threshold_percentile is None:
            threshold_percentile = self.threshold_percentile

        n_samples = self.input_tensor_scaled.shape[0]
        X = self.input_tensor_scaled.reshape(n_samples, -1)

        self.model = self._init_model()
        self.model.fit(X)

        self.anomaly_scores = self.model.decision_scores_
        self.threshold_scores = np.percentile(self.anomaly_scores, threshold_percentile)

    # --------------------------------------------------
    # Stats Output
    # --------------------------------------------------
    def get_anomaly_stats(self):
        """
        Return anomaly scores and labels per slice (1 row per slice_id).

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['sn', slice_col, 'anomaly_score', 'is_outlier']
        """
        if self.anomaly_scores is None:
            raise ValueError("Call fit() first.")

        is_outlier = self.anomaly_scores > self.threshold_scores

        unique_slices = self.df_raw[[self.slice_col]].drop_duplicates().reset_index(drop=True)
        result_df = unique_slices.copy()
        result_df["anomaly_score"] = self.anomaly_scores
        result_df["is_outlier"] = is_outlier
        result_df["sn"] = result_df[self.slice_col].apply(lambda x: str(x).split("_")[0])

        return result_df[["sn", self.slice_col, "anomaly_score", "is_outlier"]]

    # --------------------------------------------------
    # Plot Functions (return figure object)
    # --------------------------------------------------
    def plot_anomaly_score_distribution(
        self, bins=30, sample_size=10000, random_state=42
    ):
        if self.anomaly_scores is None:
            raise ValueError("Call fit() before plotting anomaly scores.")

        scores = self.anomaly_scores
        if len(scores) > sample_size:
            np.random.seed(random_state)
            scores = np.random.choice(scores, sample_size, replace=False)

        fig = plt.figure(figsize=(10, 5))
        plt.hist(scores, bins=bins, edgecolor="black", alpha=0.8)
        plt.axvline(
            self.threshold_scores,
            color="red",
            linestyle="--",
            label=f"Threshold = {self.threshold_scores:.4f}",
        )
        plt.title("Anomaly Score Distribution")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return fig

    def plot_time_series_by_category(
        self,
        category: Union[str, Tuple[float, float]] = "normal",
        n_samples=100,
        random_state=42,
    ):
        if self.anomaly_scores is None:
            raise ValueError("Call fit() before plotting.")

        np.random.seed(random_state)
        scores = self.anomaly_scores

        if category == "normal":
            idx = np.where(scores <= self.threshold_scores)[0]
        elif category == "abnormal":
            idx = np.where(scores > self.threshold_scores)[0]
        elif isinstance(category, tuple):
            lo, hi = category
            idx = np.where((scores >= lo) & (scores <= hi))[0]
        else:
            raise ValueError("Invalid category value.")

        if len(idx) == 0:
            raise ValueError("No samples match the category / range.")

        selected = np.random.choice(idx, min(n_samples, len(idx)), replace=False)
        samples = self.input_tensor[selected, :, 0]

        fig = plt.figure(figsize=(12, 5))
        cmap = cm.get_cmap("viridis", len(samples))

        for i, series in enumerate(samples):
            plt.plot(series, color=cmap(i), alpha=0.5)

        if isinstance(category, tuple):
            title = f"Time Series (score ∈ [{category[0]}, {category[1]}])"
        else:
            title = f"{category.capitalize()} Time Series"

        plt.title(title)
        plt.xlabel("Time Index")
        plt.ylabel(self.feature)
        plt.grid(True)
        plt.tight_layout()

        return fig

    def plot_mean_and_spread(self, n_samples=100, random_state=42, show_percentile=True):
        if self.anomaly_scores is None:
            raise ValueError("Call fit() before plotting.")

        np.random.seed(random_state)
        is_outlier = self.anomaly_scores > self.threshold_scores

        normal_idx = np.where(~is_outlier)[0]
        abnormal_idx = np.where(is_outlier)[0]

        normal = self.input_tensor[
            np.random.choice(normal_idx, min(n_samples, len(normal_idx)), replace=False),
            :, 0,
        ]
        abnormal = self.input_tensor[
            np.random.choice(abnormal_idx, min(n_samples, len(abnormal_idx)), replace=False),
            :, 0,
        ]

        def plot_stat(ax, data, title, color):
            mean = data.mean(axis=0)
            ax.plot(mean, color=color, label="Mean")

            if show_percentile:
                p10 = np.percentile(data, 10, axis=0)
                p90 = np.percentile(data, 90, axis=0)
                ax.fill_between(
                    np.arange(len(mean)),
                    p10,
                    p90,
                    color=color,
                    alpha=0.3,
                    label="10-90 Percentile",
                )
            else:
                std = data.std(axis=0)
                ax.fill_between(
                    np.arange(len(mean)),
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.3,
                    label="±1 Std Dev",
                )

            ax.set_title(title)
            ax.grid(True)
            ax.legend()

        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        plot_stat(axs[0], normal, "Normal Mean ± Spread", "blue")
        plot_stat(axs[1], abnormal, "Abnormal Mean ± Spread", "red")

        axs[1].set_xlabel("Time Index")
        plt.tight_layout()

        return fig

    # --------------------------------------------------
    # MLflow Helpers
    # --------------------------------------------------
    def get_params(self):
        return {
            "time_col": self.time_col,
            "feature": self.feature,
            "slice_col": self.slice_col,
            "scaler_type": self.scaler_type,
            "threshold_percentile": self.threshold_percentile,
            **(self.model_params or {}),
        }

    def save_model(self, path: str):
        joblib.dump(
            {
                "scaler": self.scaler,
                "model": self.model,
                "threshold": self.threshold_scores,
            },
            path,
        )

    @staticmethod
    def load_model(path: str):
        return joblib.load(path)
