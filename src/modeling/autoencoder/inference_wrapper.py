# inference_wrapper.py
# MLflow PyFunc wrapper for Autoencoder model, tells MLflow how to load the model and make predictions
# MLflow PyFunc requires implementing load_context and predict methods
"""
1. MLflow has a standard interface:

class PythonModel:
    def load_context(self, context):
        ...

    def predict(self, context, model_input):
        ...
We implement this interface to wrap our Autoencoder model.
"""

import mlflow.pyfunc
import numpy as np
import joblib


class AutoencoderWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for our Autoencoder.
    Loads autoencoder.pkl (scaler + model).
    Expects input: a 3D numpy array (n_samples, timesteps, 1)
    """

    def load_context(self, context):
        path = context.artifacts["autoencoder"]
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        self.threshold = obj["threshold"]

    def predict(self, context, input_tensor_3d):
        """
        Returns anomaly scores for new data.
        input_tensor_3d: numpy array (n_samples, T, 1)
        """

        # Scale if required
        if self.scaler:
            flat = input_tensor_3d.reshape(-1, input_tensor_3d.shape[-1])
            flat_scaled = self.scaler.transform(flat)
            X_scaled = flat_scaled.reshape(input_tensor_3d.shape)
        else:
            X_scaled = input_tensor_3d

        n_samples = X_scaled.shape[0]
        X_flat = X_scaled.reshape(n_samples, -1)

        scores = self.model.decision_function(X_flat)
        is_outlier = scores > self.threshold

        return {
            "scores": scores,
            "is_outlier": is_outlier,
        }
