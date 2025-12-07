# autoencoder.py

import numpy as np
import pandas as pd
import joblib
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

        scaled_flat = self.scaler.fit_transform(flat_X)
        return scaled_flat.reshape(X.shape)

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

    def predict_scores(self, new_tensor_3d: np.ndarray) -> np.ndarray:
        """
        Inference for new data. Returns anomaly scores only.
        """

        if self.scaler:
            flat = new_tensor_3d.reshape(-1, new_tensor_3d.shape[-1])
            flat_scaled = self.scaler.transform(flat)
            X_scaled = flat_scaled.reshape(new_tensor_3d.shape)
        else:
            X_scaled = new_tensor_3d

        n_samples = X_scaled.shape[0]
        X_flat = X_scaled.reshape(n_samples, -1)

        return self.model.decision_function(X_flat)

    # --------------------------------------------------
    # Stats Output
    # --------------------------------------------------
    def get_anomaly_stats(self):
        if self.anomaly_scores is None:
            raise ValueError("Call fit() first.")

        is_outlier = self.anomaly_scores > self.threshold_scores

        unique_slices = self.df_raw[[self.slice_col]].drop_duplicates().reset_index(
            drop=True
        )
        result_df = unique_slices.copy()
        result_df["anomaly_score"] = self.anomaly_scores
        result_df["is_outlier"] = is_outlier
        result_df["sn"] = result_df[self.slice_col].apply(lambda x: str(x).split("_")[0])

        return result_df[["sn", self.slice_col, "anomaly_score", "is_outlier"]]

    # --------------------------------------------------
    # MLflow Helper Functions
    # --------------------------------------------------
    def get_params(self):
        """Return parameters for MLflow logging."""
        return {
            "time_col": self.time_col,
            "feature": self.feature,
            "slice_col": self.slice_col,
            "scaler_type": self.scaler_type,
            "threshold_percentile": self.threshold_percentile,
            **(self.model_params or {}),
        }

    def save_model(self, path: str):
        """Save model + scaler using joblib."""
        obj = {
            "scaler": self.scaler,
            "model": self.model,
            "threshold": self.threshold_scores,
        }
        joblib.dump(obj, path)

    @staticmethod
    def load_model(path: str):
        """Load a trained model (for inference)."""
        return joblib.load(path)
