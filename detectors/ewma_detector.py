import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class EWMAAnomalyDetector:
    """
    EWMA-based anomaly detector with optional scaling and flexible recent window evaluation.

    Parameters:
        df (pd.DataFrame): Input time series data.
        feature (str): Target feature to detect anomalies on.
        recent_window_size (int or str): 'all' or integer; number of recent points to evaluate in scoring.
        window (int): Span for EWMA and rolling std.
        no_of_stds (float): Control limit multiplier.
        n_shift (int): Shift to prevent leakage.
        anomaly_direction (str): One of {'both', 'high', 'low'}.
        scaler (str or object): Optional scaler: 'standard', 'minmax', or custom scaler with fit_transform and inverse_transform.
        min_std_ratio (float): Minimum rolling std as a ratio of |feature| to avoid near-zero std (default: 0.01).
    """

    def __init__(
        self,
        df,
        feature,
        timestamp_col="time",
        recent_window_size="all",
        window=100,
        no_of_stds=3.0,
        n_shift=1,
        anomaly_direction="low",
        scaler=None,
        min_std_ratio=0.01,
        use_weighted_std=False
    ):
        assert anomaly_direction in {"both", "high", "low"}
        assert scaler in {None, "standard", "minmax"} or hasattr(scaler, "fit_transform")
        assert isinstance(recent_window_size, (int, type(None), str))

        self.df_original = df.copy()
        self.feature = feature
        self.timestamp_col = timestamp_col
        self.window = window
        self.no_of_stds = no_of_stds
        self.n_shift = n_shift
        self.recent_window_size = recent_window_size
        self.anomaly_direction = anomaly_direction
        self.df_ = None
        self.scaler_type = scaler
        self._scaler = None
        self.min_std_ratio = min_std_ratio
        self.use_weighted_std = use_weighted_std

    def _apply_scaler(self, df):
        df = df.copy()
        if self.scaler_type is None:
            df['feature_scaled'] = df[self.feature]
        else:
            if self.scaler_type == "standard":
                self._scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self._scaler = MinMaxScaler()
            else:
                self._scaler = self.scaler_type
            df['feature_scaled'] = self._scaler.fit_transform(df[[self.feature]])
        return df

    def _inverse_scaler(self, series):
        if self._scaler is None:
            return series
        return self._scaler.inverse_transform(series.values.reshape(-1, 1)).flatten()

    def _weighted_std_ewm(self, series, span):
        """
        Calculate exponentially weighted standard deviation.

        Formula:
            σ_w = sqrt( Σ wᵢ (xᵢ - μ_w)² / Σ wᵢ )

        Where:
            - xᵢ: input values in the rolling window
            - wᵢ: exponential weights (more recent points have higher weight)
            - μ_w: weighted mean = Σ wᵢ xᵢ / Σ wᵢ

        Parameters:
            series (pd.Series): Input series to compute weighted std on
            span (int): EWMA span (same as for EMA)

        Returns:
            pd.Series: weighted std aligned with EMA
        """
        import numpy as np
        alpha = 2 / (span + 1)
        weights = np.array([(1 - alpha) ** i for i in reversed(range(span))])
        weights /= weights.sum()

        x = series.values
        stds = []
        for i in range(len(x)):
            if i < span:
                stds.append(np.nan)
            else:
                window = x[i - span + 1:i + 1]
                mu_w = np.sum(weights * window)
                var_w = np.sum(weights * (window - mu_w) ** 2)
                stds.append(np.sqrt(var_w))
        return pd.Series(stds, index=series.index)


    def _add_ewma(self):
        
        df = self._apply_scaler(self.df_original)

        target = df['feature_scaled'].shift(self.n_shift)
        
        df['EMA'] = target.ewm(span=self.window, adjust=False).mean()
        if self.use_weighted_std:
            df['rolling_std'] = self._weighted_std_ewm(target, span=self.window)
        else:
            df['rolling_std'] = target.rolling(window=self.window).std()
        
        # Impose a lower bound on std to avoid degenerate control limits
        min_std = self.min_std_ratio * df['feature_scaled'].abs()
        df['rolling_std'] = df['rolling_std'].where(df['rolling_std'] >= min_std, min_std)

        df['UCL'] = df['EMA'] + self.no_of_stds * df['rolling_std']
        df['LCL'] = df['EMA'] - self.no_of_stds * df['rolling_std']
        
        return df

    def _detect_anomalies(self, df):
        if self.anomaly_direction == "high":
            df['is_outlier'] = df['feature_scaled'] > df['UCL']
        elif self.anomaly_direction == "low":
            df['is_outlier'] = df['feature_scaled'] < df['LCL']
        else:
            df['is_outlier'] = (df['feature_scaled'] > df['UCL']) | (df['feature_scaled'] < df['LCL'])
        
        df.loc[df.index[:self.window], 'is_outlier'] = False
        
        return df

    def fit(self):
        df = self._add_ewma()
        df = self._detect_anomalies(df)
        df_clean = df.dropna(subset=["EMA", "UCL", "LCL", "feature_scaled"])

        if self.recent_window_size in [None, "all"]:
            recent_df = df_clean
        else:
            recent_df = df_clean.tail(self.recent_window_size)

        self.df_ = df
        return recent_df[recent_df["is_outlier"]][["sn", self.timestamp_col, self.feature, "is_outlier"]]
