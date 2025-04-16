import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Class for detecting and handling anomalies in time series data"""

    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05):
        """
        Initialize anomaly detector

        Args:
            method: Anomaly detection method
                - 'isolation_forest': Isolation Forest algorithm
                - 'lof': Local Outlier Factor algorithm
                - 'one_class_svm': One-Class SVM algorithm
            contamination: Expected proportion of outliers in the data
        """
        self.method = method
        self.contamination = contamination
        self.detector = None
        self.scaler = StandardScaler()
        self.anomaly_indices = None
        self.anomaly_scores = None

    def fit_detect(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit detector and detect anomalies

        Args:
            X: Input features

        Returns:
            Binary array where 1 indicates an anomaly
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit detector
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'lof':
            self.detector = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")

        # Fit detector
        self.detector.fit(X_scaled)

        # Predict anomalies
        if self.method == 'lof':
            # LOF returns -1 for outliers, 1 for inliers
            predictions = self.detector.predict(X_scaled)
            anomalies = np.where(predictions == -1, 1, 0)

            # Get anomaly scores
            self.anomaly_scores = -self.detector.decision_function(X_scaled)
        else:
            # Isolation Forest and One-Class SVM return -1 for outliers, 1 for inliers
            predictions = self.detector.predict(X_scaled)
            anomalies = np.where(predictions == -1, 1, 0)

            # Get anomaly scores
            self.anomaly_scores = -self.detector.decision_function(X_scaled)

        # Store anomaly indices
        self.anomaly_indices = np.where(anomalies == 1)[0]

        return anomalies

    def get_anomaly_indices(self) -> np.ndarray:
        """
        Get indices of detected anomalies

        Returns:
            Array of anomaly indices
        """
        if self.anomaly_indices is None:
            raise ValueError("No anomalies detected. Call fit_detect() first.")

        return self.anomaly_indices

    def get_anomaly_scores(self) -> np.ndarray:
        """
        Get anomaly scores

        Returns:
            Array of anomaly scores
        """
        if self.anomaly_scores is None:
            raise ValueError("No anomaly scores available. Call fit_detect() first.")

        return self.anomaly_scores

    def detect_and_handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle anomalies in a DataFrame

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled anomalies
        """
        logger.info(f"Detecting anomalies using method: {self.method}")

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Check if DataFrame is empty
        if df_copy.empty:
            logger.warning("Empty DataFrame provided to anomaly detection")
            return df_copy

        # Ensure timestamp column is preserved
        has_timestamp = 'timestamp' in df_copy.columns
        timestamp_col = None
        if has_timestamp:
            timestamp_col = df_copy['timestamp'].copy()
            df_copy = df_copy.drop(columns=['timestamp'])

        try:
            # Select only numeric columns for anomaly detection
            numeric_df = df_copy.select_dtypes(include=[np.number])

            # Check if we have enough numeric columns
            if numeric_df.shape[1] == 0:
                logger.warning("No numeric columns found for anomaly detection")
                # Restore timestamp column if it existed
                if has_timestamp:
                    df_copy['timestamp'] = timestamp_col
                return df_copy

            # Detect anomalies
            anomalies = self.fit_detect(numeric_df.values)

            # Get indices of anomalies
            anomaly_indices = np.where(anomalies == 1)[0]

            if len(anomaly_indices) == 0:
                logger.info("No anomalies detected")
                # Restore timestamp column if it existed
                if has_timestamp:
                    df_copy['timestamp'] = timestamp_col
                return df_copy

            logger.info(f"Detected {len(anomaly_indices)} anomalies")

            # Handle anomalies by replacing with interpolated values
            df_handled = df_copy.copy()

            # Mark anomalies
            df_handled['is_anomaly'] = 0
            df_handled.loc[anomaly_indices, 'is_anomaly'] = 1

            # Interpolate anomalies
            for col in numeric_df.columns:
                # Create a mask for anomalies in this column
                mask = df_handled['is_anomaly'] == 1

                # Skip if no anomalies or column is all NaN
                if not mask.any() or df_handled[col].isna().all():
                    continue

                # Store original values for logging
                original_values = df_handled.loc[mask, col].copy()

                # Interpolate anomalies
                df_handled.loc[mask, col] = np.nan
                df_handled[col] = df_handled[col].interpolate(method='linear', limit_direction='both')

                # If there are still NaNs (e.g., at the beginning or end), use forward/backward fill
                if df_handled[col].isna().any():
                    df_handled[col] = df_handled[col].ffill().bfill()

                # Log changes
                new_values = df_handled.loc[mask, col]
                logger.debug(f"Replaced anomalies in column {col}: {original_values.values} -> {new_values.values}")

            # Remove the is_anomaly column
            df_handled = df_handled.drop(columns=['is_anomaly'])

            # Restore timestamp column if it existed
            if has_timestamp:
                df_handled['timestamp'] = timestamp_col

            logger.info(f"Successfully handled {len(anomaly_indices)} anomalies")
            return df_handled

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            logger.error(f"Returning original DataFrame")

            # Restore timestamp column if it existed
            if has_timestamp:
                df_copy['timestamp'] = timestamp_col

            return df_copy

    def plot_anomalies(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """
        Plot anomalies

        Args:
            X: Input features
            feature_names: Feature names (required if X is numpy array)
            save_path: Path to save the plot (optional)
        """
        if self.anomaly_indices is None:
            raise ValueError("No anomalies detected. Call fit_detect() first.")

        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        elif feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Create DataFrame for plotting
        df = pd.DataFrame(X, columns=feature_names)
        df['anomaly'] = 0
        df.loc[self.anomaly_indices, 'anomaly'] = 1
        df['anomaly_score'] = 0
        df['anomaly_score'] = self.anomaly_scores

        # Plot
        plt.figure(figsize=(15, 10))

        # Plot anomaly scores
        plt.subplot(2, 1, 1)
        plt.plot(df['anomaly_score'], label='Anomaly Score')
        plt.scatter(
            df.index[df['anomaly'] == 1],
            df['anomaly_score'][df['anomaly'] == 1],
            color='red',
            label='Anomaly'
        )
        plt.title('Anomaly Scores')
        plt.legend()

        # Plot feature heatmap
        plt.subplot(2, 1, 2)

        # Select a subset of features if there are too many
        if len(feature_names) > 10:
            # Select top 10 features with highest variance
            feature_vars = np.var(X, axis=0)
            top_features = np.argsort(feature_vars)[-10:]
            selected_features = [feature_names[i] for i in top_features]
        else:
            selected_features = feature_names

        # Create heatmap data
        heatmap_data = df[selected_features].copy()

        # Normalize data for better visualization
        for col in heatmap_data.columns:
            heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].mean()) / heatmap_data[col].std()

        # Plot heatmap
        sns.heatmap(
            heatmap_data.T,
            cmap='viridis',
            cbar_kws={'label': 'Normalized Value'}
        )
        plt.title('Feature Heatmap')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def remove_anomalies(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Remove detected anomalies from data

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Tuple of (X_clean, y_clean) with anomalies removed
        """
        if self.anomaly_indices is None:
            raise ValueError("No anomalies detected. Call fit_detect() first.")

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Create mask for non-anomalous data
        mask = np.ones(X.shape[0], dtype=bool)
        mask[self.anomaly_indices] = False

        # Filter data
        X_clean = X[mask]

        # Filter target if provided
        if y is not None:
            y_clean = y[mask]
            return X_clean, y_clean
        else:
            return X_clean, None

    def interpolate_anomalies(self, X: Union[np.ndarray, pd.DataFrame], method: str = 'linear') -> np.ndarray:
        """
        Interpolate anomalies instead of removing them

        Args:
            X: Input features
            method: Interpolation method ('linear', 'nearest', 'cubic', 'spline')

        Returns:
            Data with anomalies interpolated
        """
        if self.anomaly_indices is None:
            raise ValueError("No anomalies detected. Call fit_detect() first.")

        # Convert to DataFrame for interpolation
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # Create a copy for interpolation
        df_interp = df.copy()

        # Mark anomalies as NaN
        df_interp.loc[self.anomaly_indices] = np.nan

        # Interpolate
        df_interp = df_interp.interpolate(method=method, axis=0)

        # Fill any remaining NaNs (at the beginning or end)
        df_interp = df_interp.ffill().bfill()

        # Return as numpy array if input was numpy array
        if isinstance(X, np.ndarray):
            return df_interp.values
        else:
            return df_interp


class TimeSeriesAnomalyDetector(AnomalyDetector):
    """Specialized anomaly detector for time series data"""

    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05, window_size: int = 10):
        """
        Initialize time series anomaly detector

        Args:
            method: Anomaly detection method
            contamination: Expected proportion of outliers in the data
            window_size: Size of sliding window for feature extraction
        """
        super().__init__(method, contamination)
        self.window_size = window_size

    def extract_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Extract time series features

        Args:
            X: Input time series data

        Returns:
            Extracted features
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Not enough data for feature extraction
        if n_samples < self.window_size:
            return X

        # Extract features
        features = []

        for i in range(n_samples - self.window_size + 1):
            window = X[i:i+self.window_size]

            # Calculate statistics for each feature
            window_features = []

            for j in range(n_features):
                feature_window = window[:, j]

                # Basic statistics
                mean = np.mean(feature_window)
                std = np.std(feature_window)
                min_val = np.min(feature_window)
                max_val = np.max(feature_window)

                # Trend (simple linear regression)
                x = np.arange(self.window_size)
                trend = np.polyfit(x, feature_window, 1)[0]

                # Add to window features
                window_features.extend([mean, std, min_val, max_val, trend])

            features.append(window_features)

        return np.array(features)

    def fit_detect(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit detector and detect anomalies in time series data

        Args:
            X: Input time series data

        Returns:
            Binary array where 1 indicates an anomaly
        """
        # Extract features
        X_features = self.extract_features(X)

        # Call parent fit_detect method
        anomalies = super().fit_detect(X_features)

        # Adjust anomaly indices to match original data
        if self.anomaly_indices is not None:
            self.anomaly_indices = self.anomaly_indices + self.window_size - 1

        # Adjust anomaly scores to match original data
        if self.anomaly_scores is not None:
            # Pad with zeros at the beginning
            padded_scores = np.zeros(X.shape[0])
            padded_scores[self.window_size-1:] = self.anomaly_scores
            self.anomaly_scores = padded_scores

        # Create full anomaly array
        full_anomalies = np.zeros(X.shape[0])
        full_anomalies[self.window_size-1:] = anomalies

        return full_anomalies
