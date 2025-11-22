from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ARIMAAnomalyDetectorSM:
    """
    Anomaly detection for univariate time series using statsmodels ARIMA with prediction intervals.
    Includes train/test split for evaluation and gap filling with user-specified frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the time column and feature column.
    time_col : str
        Name of the timestamp column.
    feature : str
        Name of the feature column (numeric series).
    order : tuple
        ARIMA (p,d,q) order.
    confidence_level : float
        Confidence interval (default=0.99).
    anomaly_direction : str
        "both", "upper", or "lower" anomalies to detect.
    horizon : int
        Forecast horizon = size of the test set.
    freq : str
        Frequency string for resampling (e.g. "H", "D", "15T").
    fill_method : str
        Method to fill gaps: "interpolate", "ffill", "bfill", "zero".
    """

    def __init__(self, df, time_col, feature, order=(1,1,1), 
                 confidence_level=0.99, anomaly_direction='lower', 
                 horizon=24, freq="h", fill_method="interpolate"):

        self.df = df.copy()
        self.time_col = time_col
        self.feature = feature
        self.order = order
        self.confidence_level = confidence_level
        self.anomaly_direction = anomaly_direction
        self.horizon = horizon
        self.freq = freq
        self.fill_method = fill_method

        # Containers
        self.train_df = None
        self.test_df = None
        self.results = None
        self.forecast_df = None
        self.final_result = None

    def prepare_data(self):
        df_arima = self.df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima = df_arima.set_index("ds")
        df_arima.index = pd.to_datetime(df_arima.index)

        df_arima = df_arima.resample(self.freq).mean()

        # 1. Reindex with explicit frequency to fill missing timestamps
        full_index = pd.date_range(start=df_arima.index.min(),
                                   end=df_arima.index.max(),
                                   freq=self.freq)
        df_arima = df_arima.reindex(full_index)

        # 2. Fill missing values
        if self.fill_method == "interpolate":
            df_arima['y'] = df_arima['y'].interpolate(method="linear")
        elif self.fill_method == "ffill":
            df_arima['y'] = df_arima['y'].ffill()
        elif self.fill_method == "bfill":
            df_arima['y'] = df_arima['y'].bfill()
        elif self.fill_method == "zero":
            df_arima['y'] = df_arima['y'].fillna(0)
        else:
            raise ValueError(f"Unknown fill_method: {self.fill_method}")

        # Train/test split
        split_idx = len(df_arima) - self.horizon
        self.train_df = df_arima.iloc[:split_idx]
        self.test_df = df_arima.iloc[split_idx:]

    def fit_forecast(self):
        self.model = ARIMA(self.train_df['y'], order=self.order)
        self.results = self.model.fit()

        # Forecast horizon steps ahead
        forecast_res = self.results.get_forecast(steps=self.horizon)
        forecast_ci = forecast_res.conf_int(alpha=1-self.confidence_level)

        self.forecast_df = pd.DataFrame({
            'ds': self.test_df.index,
            'forecast': forecast_res.predicted_mean.values,
            'lo': forecast_ci.iloc[:,0].values,
            'hi': forecast_ci.iloc[:,1].values,
            'actual': self.test_df['y'].values
        })

    def detect_anomalies(self):
        df = self.forecast_df.copy()

        if self.anomaly_direction == 'lower':
            df['anomaly'] = df['actual'] < df['lo']
            df['anomaly_type'] = df['anomaly'].map(lambda x: 'low' if x else None)
        elif self.anomaly_direction == 'upper':
            df['anomaly'] = df['actual'] > df['hi']
            df['anomaly_type'] = df['anomaly'].map(lambda x: 'high' if x else None)
        else:
            is_low = df['actual'] < df['lo']
            is_high = df['actual'] > df['hi']
            df['anomaly'] = is_low | is_high
            df['anomaly_type'] = is_low.map({True: 'low'}).combine_first(is_high.map({True: 'high'}))
        df['ds'] = pd.to_datetime(self.df[self.time_col].iloc[-self.horizon:].values)
        df['sn'] = self.df['sn'].iloc[0]
        self.final_result = df

    def plot_results(self):
        plt.figure(figsize=(16,6))

        # Plot training data
        plt.plot(self.train_df.index, self.train_df['y'], label="Training Data")

        # Plot test actuals
        plt.plot(self.test_df.index, self.test_df['y'], label="Test Actual", color='black')

        # Plot forecast
        plt.plot(self.final_result['ds'], self.final_result['forecast'], label="Forecast", color='orange')

        # Confidence interval
        plt.fill_between(self.final_result['ds'], 
                         self.final_result['lo'], 
                         self.final_result['hi'], 
                         color='gray', alpha=0.3, label="Confidence Interval")

        # Highlight anomalies
        low = self.final_result[self.final_result['anomaly_type']=='low']
        high = self.final_result[self.final_result['anomaly_type']=='high']
        plt.scatter(low['ds'], low['actual'], color='blue', label='Low Anomalies')
        plt.scatter(high['ds'], high['actual'], color='red', label='High Anomalies')

        plt.title(f"ARIMA Anomaly Detection (horizon={self.horizon})")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.legend()
        plt.show()

    def run(self):
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()
        return self.final_result


class ARIMAAnomalyDetectorFuture:
    """
    Detect anomalies by forecasting *future* values outside prediction intervals.

    Parameters
    ----------
    df : pandas.DataFrame
    time_col : str
    feature : str
        Column name of the numeric series to model.
    season_length : int, default=24
        Seasonal period passed to `AutoARIMA` (e.g., 24 for hourly data with daily seasonality).
    confidence_level : int, default=99
    freq : str, default='h'
        Pandas/StatsForecast frequency alias (e.g., 'h' for hourly, 'D' for
        daily). Must match the cadence of `time_col`.
    anomaly_direction : {'both', 'upper', 'lower'}, default='both'
        Which side(s) of the interval to treat as anomalous:
        - 'both' : flag values below the lower bound or above the upper bound
        - 'upper': flag values strictly above the upper bound
        - 'lower': flag values strictly below the lower bound
    split_idx : int, default=24
        Number of trailing observations reserved for the test (forecast) horizon.
        The model is trained on all rows except the last `split_idx`, then
        forecasts `h=split_idx` steps ahead for comparison against the held-out
        actuals.

    Attributes
    ----------
    model : StatsForecast
        The StatsForecast object configured with an `AutoARIMA` model.
    train_df : pandas.DataFrame or None
        Prepared training data with columns ['unique_id', 'ds', 'y'].
    test_df : pandas.DataFrame or None
        Prepared test data (last `split_idx` rows) with columns
        ['unique_id', 'ds', 'y'].
    forecast_df : pandas.DataFrame or None
        Forecast output from StatsForecast, including point forecasts in
        'AutoARIMA' and interval columns named
        'AutoARIMA-lo-{confidence_level}' and 'AutoARIMA-hi-{confidence_level}'.
    result_df : pandas.DataFrame or None
        Merge of `forecast_df` and `test_df` with anomaly flags:
        - 'anomaly' (bool)
        - 'anomaly_type' in {'low', 'high', None}

    Methods
    -------
    prepare_data()
        Renames/standardizes columns to ['unique_id', 'ds', 'y'] and splits
        into train/test by `split_idx`.
    fit_forecast()
        Fits AutoARIMA on the training data and produces `h=split_idx`
        forecasts with prediction intervals at `confidence_level`.
    detect_anomalies()
        Compares actuals to forecast intervals and sets 'anomaly' and
        'anomaly_type' columns in `result_df`.
    plot_anomalies()
        Plots actuals, forecasts, prediction interval, and highlights detected
        anomalies.
    run()
        Convenience method: `prepare_data()` → `fit_forecast()` → `detect_anomalies()`.

    Examples
    --------
    >>> detector = ARIMAAnomalyDetectorFuture(
    ...     df=dataframe,
    ...     time_col="timestamp",
    ...     feature="throughput",
    ...     season_length=24,
    ...     confidence_level=95,
    ...     freq="h",
    ...     anomaly_direction="both",
    ...     split_idx=48,
    ... )
    >>> detector.run()
    >>> anomalies = detector.result_df[detector.result_df["anomaly"]]
    >>> detector.plot_anomalies()
    """

    def __init__(self, df, time_col, feature, season_length=24, confidence_level=99,
                 freq='h', anomaly_direction='both', split_idx=1, unique_id = "series_1"):
        self.df = df
        self.time_col = time_col
        self.feature = feature
        self.season_length = season_length
        self.confidence_level = confidence_level
        self.freq = freq
        self.anomaly_direction = anomaly_direction
        self.split_idx = split_idx
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=season_length)],
            freq=freq,
            n_jobs=-1
        )
        self.unique_id = unique_id
        self.train_df = None
        self.test_df = None
        self.df_arima = None
        self.forecast_df = None

    def prepare_data(self):
        df_arima = self.df[[self.time_col, self.feature]].copy()
        df_arima = df_arima.rename(columns={self.time_col: "ds", self.feature: "y"})
        df_arima["unique_id"] = self.unique_id
        self.df_arima = df_arima
        # Split train and test
        self.train_df = df_arima[:-self.split_idx].copy()
        self.test_df = df_arima[-self.split_idx:].copy()

    def fit_forecast(self):
        self.forecast_df = self.model.forecast(
            df=self.train_df,
            h=self.split_idx,
            level=[self.confidence_level]
        ).reset_index()

    def detect_anomalies(self):
        # Merge forecast and actuals
        result = pd.merge(
            self.forecast_df.drop(columns='ds'),
            self.test_df,
            on=["unique_id"],
            how="left"
        )
        lo_col = f"AutoARIMA-lo-{self.confidence_level}"
        hi_col = f"AutoARIMA-hi-{self.confidence_level}"

        if self.anomaly_direction == "lower":
            result["anomaly"] = result["y"] < result[lo_col]
            result["anomaly_type"] = result["anomaly"].apply(lambda x: "low" if x else None)
        elif self.anomaly_direction == "upper":
            result["anomaly"] = result["y"] > result[hi_col]
            result["anomaly_type"] = result["anomaly"].apply(lambda x: "high" if x else None)
        else:
            is_low = result["y"] < result[lo_col]
            is_high = result["y"] > result[hi_col]
            result["anomaly"] = is_low | is_high
            result["anomaly_type"] = is_low.map({True: "low"}).combine_first(is_high.map({True: "high"}))

        self.result_df = result
        
    def plot_anomalies(self):
        """
        Plot:
        - Train actuals (line) from self.train_df
        - Forecast horizon from self.result_df:
            * Forecast (scatter)
            * Actual (scatter, different color)
            * CI bounds (scatter + connected lines)
            * Shaded CI band (handles single-point horizon)
        """
        if self.train_df is None or getattr(self, 'result_df', None) is None:
            raise RuntimeError("Call run() before plotting.")

        lo_col = f"AutoARIMA-lo-{self.confidence_level}"
        hi_col = f"AutoARIMA-hi-{self.confidence_level}"

        # Train (history)
        train = self.train_df[['ds', 'y']].sort_values('ds')

        # Horizon with forecast + CI (already merged in detect_anomalies)
        need = ['ds', 'y', 'AutoARIMA', lo_col, hi_col]
        missing = [c for c in need if c not in self.result_df.columns]
        if missing:
            raise ValueError(f"Missing columns in result_df: {missing}")

        res = (
            self.result_df[need]
            .dropna(subset=['ds'])
            .drop_duplicates('ds')
            .sort_values('ds')
        )

        plt.figure(figsize=(16, 5))

        # 1) Train actuals (line)
        plt.plot(self.df_arima['ds'], self.df_arima['y'], label='Train Actual', linewidth=1.2, alpha=0.85,  color='blue')

        dark_color = 'darkorange'
        light_color = 'orange'  # A lighter shade of blue

        # 2) Forecast vs Actual on horizon (scatter, different colors)
        plt.scatter(res['ds'], res['AutoARIMA'], s=30, label='Forecast',  color=dark_color)
        plt.scatter(res['ds'], res['y'], s=30, label='Actual (Horizon)',  color='blue')

        # 3) Confidence interval bounds (scatter + connected lines), plus shaded band
        if len(res['ds']) > 1:
            plt.plot(res['ds'], res[lo_col], linewidth=1.0, label=f'CI Low ({self.confidence_level}%)',  color=light_color)
            plt.plot(res['ds'], res[hi_col], linewidth=1.0, label=f'CI High ({self.confidence_level}%)',  color=light_color)
            plt.scatter(res['ds'], res[lo_col], s=14,  color=light_color)
            plt.scatter(res['ds'], res[hi_col], s=14,  color=light_color)
            plt.fill_between(res['ds'], res[lo_col], res[hi_col], alpha=1, label=f'{self.confidence_level}% CI')

        else:
            # Plot the two scatter points
            plt.scatter(res['ds'], res[lo_col], s=14, label=f'CI Low ({self.confidence_level}%)',  color=light_color)
            plt.scatter(res['ds'], res[hi_col], s=14, label=f'CI High ({self.confidence_level}%)',  color=light_color)

            # Connect the two points with a transparent line
            plt.plot([res['ds'].iloc[0], res['ds'].iloc[0]], 
                    [res[lo_col].iloc[0], res[hi_col].iloc[0]], 
                    color=light_color, 
                    alpha=0.3, 
                    linestyle='--')


        # 4) Optional: highlight anomalies (horizon)
        if 'anomaly' in self.result_df.columns and self.result_df['anomaly'].fillna(False).any():
            a = self.result_df[self.result_df['anomaly'].fillna(False)]
            plt.scatter(a['ds'], a['y'], s=60, facecolors='none', edgecolors='red', linewidths=1.2, label='Anomaly')

        plt.title(f"ARIMA Anomaly Detection (Train & Horizon, {self.anomaly_direction})")
        plt.xlabel("Time")
        plt.ylabel(self.feature)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def run(self):
        self.prepare_data()
        self.fit_forecast()
        self.detect_anomalies()

