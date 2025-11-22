# =============================
# Third-Party Libraries
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
import torch
from pyod.models.auto_encoder_torch import AutoEncoder

# ======================================================================
# Classes / Functions (kept identical in content; only ordering changed)
# ======================================================================
class c:
    def __init__(self,
                 df: pd.DataFrame,
                 time_col: str,
                 feature: str,
                 slice_col: str = "slice_id",
                 model_params: Optional[dict] = None,
                 external_model: Optional[object] = None,
                 scaler: Union[str, object, None] = "None",
                 threshold_percentile: float = 99):
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

    def _build_tensor_from_slices(self):
        grouped = self.df_raw.groupby(self.slice_col)
        tensors = []

        for _, group in grouped:
            series = group.sort_values(by=self.time_col)[self.feature].values
            tensors.append(series)

        tensor_3d = np.stack(tensors)[:, :, np.newaxis]  # shape: (n_samples, n_timesteps, 1)
        return tensor_3d

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_type is None:
            return X
        flat_X = X.reshape(-1, X.shape[-1])  # flatten across time axis
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = self.scaler_type
        scaled_flat = self.scaler.fit_transform(flat_X)
        return scaled_flat.reshape(X.shape)

    def prepare(self):
        tensor = self._build_tensor_from_slices()
        self.input_tensor = tensor
        self.input_tensor_scaled = self._apply_scaler(tensor)

    def _init_model(self):
        if self.external_model:
            return self.external_model
        default_params = {
            "hidden_neurons": [self.input_tensor.shape[1], 32, 32, self.input_tensor.shape[1]],
            "hidden_activation": "relu",
            "epochs": 10,
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
        X = self.input_tensor_scaled.reshape(n_samples, -1)  # flatten to 2D for sklearn-compatible model
        self.model = self._init_model()
        self.model.fit(X)

        self.anomaly_scores = self.model.decision_scores_
        self.threshold_scores = np.percentile(self.anomaly_scores, threshold_percentile)

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

