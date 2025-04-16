import numpy as np
import pandas as pd
import traceback
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from boruta import BorutaPy

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Class for selecting important features"""

    def __init__(self, method: str = 'importance'):
        """
        Initialize feature selector

        Args:
            method: Feature selection method
                - 'importance': Feature importance from tree-based models
                - 'correlation': Correlation with target
                - 'mutual_info': Mutual information with target
                - 'boruta': Boruta algorithm
                - 'pca': Principal Component Analysis
        """
        self.method = method
        self.selector = None
        self.selected_features = None
        self.feature_names = None
        self.importance_scores = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit feature selector

        Args:
            X: Input features
            y: Target values
            feature_names: Feature names (required if X is numpy array)
        """
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Apply feature selection method
        if self.method == 'importance':
            self._importance_selection(X, y)
        elif self.method == 'correlation':
            self._correlation_selection(X, y)
        elif self.method == 'mutual_info':
            self._mutual_info_selection(X, y)
        elif self.method == 'boruta':
            self._boruta_selection(X, y)
        elif self.method == 'pca':
            self._pca_selection(X)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using selected features

        Args:
            X: Input features

        Returns:
            Transformed data
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.selector.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform data

        Args:
            X: Input features
            y: Target values
            feature_names: Feature names (required if X is numpy array)

        Returns:
            Transformed data
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def select_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Select important features from a DataFrame

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting features using method: {self.method}")

        # Check if DataFrame is empty
        if df.empty:
            logger.warning("Empty DataFrame provided to feature selection")
            return df

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Ensure timestamp column is preserved
        has_timestamp = 'timestamp' in df_copy.columns
        timestamp_col = df_copy['timestamp'].copy() if has_timestamp else None

        # Check if we have enough rows
        if len(df_copy) < 10:  # Minimum required for most feature selection methods
            logger.warning(f"Too few samples ({len(df_copy)}) for feature selection. Returning original DataFrame.")
            return df_copy

        # Separate features and target
        try:
            # First, ensure all numeric columns are properly typed
            for col in df_copy.columns:
                if col != 'timestamp' and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    try:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric: {e}")

            # Get numeric columns
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Numeric columns after conversion: {numeric_cols}")

            if target_col in df_copy.columns:
                # Convert target to numeric if needed
                if not pd.api.types.is_numeric_dtype(df_copy[target_col]):
                    df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')

                y = df_copy[target_col].values
                X = df_copy.drop(columns=['timestamp'] if has_timestamp else []).drop(columns=[target_col])
            else:
                logger.warning(f"Target column '{target_col}' not found in DataFrame. Using first numeric column as target.")
                if len(numeric_cols) == 0:
                    logger.error("No numeric columns found in DataFrame")
                    # Create synthetic columns
                    df_copy['synthetic_target'] = np.random.normal(0, 1, len(df_copy))
                    df_copy['synthetic_feature_1'] = np.random.normal(0, 1, len(df_copy))
                    df_copy['synthetic_feature_2'] = np.random.normal(0, 1, len(df_copy))

                    target_col = 'synthetic_target'
                    y = df_copy[target_col].values
                    X = df_copy[['synthetic_feature_1', 'synthetic_feature_2']]
                else:
                    target_col = numeric_cols[0]
                    y = df_copy[target_col].values
                    X = df_copy.drop(columns=['timestamp'] if has_timestamp else []).drop(columns=[target_col])
        except Exception as e:
            logger.error(f"Error separating features and target: {e}")
            logger.error(traceback.format_exc())
            # Create synthetic data as fallback
            df_copy['synthetic_target'] = np.random.normal(0, 1, len(df_copy))
            df_copy['synthetic_feature_1'] = np.random.normal(0, 1, len(df_copy))
            df_copy['synthetic_feature_2'] = np.random.normal(0, 1, len(df_copy))

            target_col = 'synthetic_target'
            y = df_copy[target_col].values
            X = df_copy[['synthetic_feature_1', 'synthetic_feature_2']]

        # Check if we have enough features
        if X.shape[1] <= 1:
            logger.warning(f"Too few features ({X.shape[1]}) for feature selection. Returning original DataFrame.")
            return df_copy

        # Check for NaN values
        try:
            # First, ensure X is a DataFrame for easier handling
            if not isinstance(X, pd.DataFrame):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=self.feature_names)
                else:
                    logger.error(f"Unexpected type for X: {type(X)}")
                    raise ValueError(f"Unexpected type for X: {type(X)}")

            # Check for NaN values in X
            if X.isna().any().any():
                logger.warning("NaN values found in X. Filling NaNs before feature selection.")
                X = X.fillna(X.mean())

            # Check for NaN values in y
            if isinstance(y, np.ndarray) and np.isnan(y).any():
                logger.warning("NaN values found in y. Filling NaNs before feature selection.")
                y = np.nan_to_num(y, nan=np.nanmean(y))
            elif isinstance(y, pd.Series) and y.isna().any():
                logger.warning("NaN values found in y. Filling NaNs before feature selection.")
                y = y.fillna(y.mean())
        except Exception as e:
            logger.error(f"Error checking for NaN values: {e}")
            logger.error(f"X type: {type(X)}, y type: {type(y)}")
            logger.error(traceback.format_exc())
            # Create dummy X and y if needed
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(np.zeros((len(y), 3)), columns=['feature_1', 'feature_2', 'feature_3'])
            if not isinstance(y, np.ndarray):
                y = np.zeros(len(X))

        try:
            # Fit and transform
            X_selected = self.fit_transform(X, y)

            # Create new DataFrame with selected features
            if isinstance(X_selected, pd.DataFrame):
                df_selected = X_selected
            else:
                # Convert numpy array back to DataFrame
                df_selected = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)

            # Add target column back
            df_selected[target_col] = y

            # Add timestamp column back if it existed
            if has_timestamp:
                df_selected['timestamp'] = timestamp_col

            logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
            return df_selected

        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            logger.error(f"Returning original DataFrame")
            return df_copy

    def get_selected_features(self) -> List[str]:
        """
        Get names of selected features

        Returns:
            List of selected feature names
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Call fit() first.")

        return self.selected_features

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.importance_scores is None:
            raise ValueError("No importance scores available. Call fit() first.")

        return dict(zip(self.feature_names, self.importance_scores))

    def plot_feature_importance(self, save_path: Optional[str] = None):
        """
        Plot feature importance

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.importance_scores is None:
            raise ValueError("No importance scores available. Call fit() first.")

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.importance_scores
        })

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def _importance_selection(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.01):
        """
        Select features based on importance from tree-based models

        Args:
            X: Input features
            y: Target values
            threshold: Importance threshold for feature selection
        """
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        self.importance_scores = rf.feature_importances_

        # Select features above threshold
        selected_indices = np.where(self.importance_scores >= threshold)[0]
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        # Create selector
        self.selector = SelectKBest(f_regression, k=len(selected_indices))
        self.selector.fit(X, y)

        logger.info(f"Selected {len(self.selected_features)} features based on importance")

    def _correlation_selection(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.1):
        """
        Select features based on correlation with target

        Args:
            X: Input features
            y: Target values
            threshold: Correlation threshold for feature selection
        """
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))  # Use absolute correlation

        self.importance_scores = np.array(correlations)

        # Select features above threshold
        selected_indices = np.where(self.importance_scores >= threshold)[0]
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        # Create selector
        self.selector = SelectKBest(f_regression, k=len(selected_indices))
        self.selector.fit(X, y)

        logger.info(f"Selected {len(self.selected_features)} features based on correlation")

    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.01):
        """
        Select features based on mutual information with target

        Args:
            X: Input features
            y: Target values
            threshold: Mutual information threshold for feature selection
        """
        # Calculate mutual information with target
        mi = mutual_info_regression(X, y)
        self.importance_scores = mi

        # Select features above threshold
        selected_indices = np.where(self.importance_scores >= threshold)[0]
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        # Create selector
        self.selector = SelectKBest(mutual_info_regression, k=len(selected_indices))
        self.selector.fit(X, y)

        logger.info(f"Selected {len(self.selected_features)} features based on mutual information")

    def _boruta_selection(self, X: np.ndarray, y: np.ndarray):
        """
        Select features using Boruta algorithm

        Args:
            X: Input features
            y: Target values
        """
        # Initialize Boruta
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)

        # Fit Boruta
        boruta.fit(X, y)

        # Get feature importance
        self.importance_scores = boruta.ranking_

        # Select confirmed features
        selected_indices = np.where(boruta.support_)[0]
        self.selected_features = [self.feature_names[i] for i in selected_indices]

        # Create selector based on selected features
        self.selector = SelectKBest(f_regression, k=len(selected_indices))
        self.selector.fit(X, y)

        logger.info(f"Selected {len(self.selected_features)} features using Boruta")

    def _pca_selection(self, X: np.ndarray, n_components: float = 0.95):
        """
        Select features using PCA

        Args:
            X: Input features
            n_components: Number of components or variance ratio to keep
        """
        # Create PCA pipeline with standardization
        self.selector = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components))
        ])

        # Fit PCA
        self.selector.fit(X)

        # Get explained variance ratio
        pca = self.selector.named_steps['pca']
        self.importance_scores = pca.explained_variance_ratio_

        # PCA doesn't select original features, so set selected_features to None
        self.selected_features = None

        logger.info(f"PCA selected {pca.n_components_} components explaining {n_components*100:.1f}% of variance")


class TimeSeriesFeatureSelector(FeatureSelector):
    """Feature selector for time series data"""

    def __init__(self, method: str = 'importance'):
        """
        Initialize time series feature selector

        Args:
            method: Feature selection method
        """
        super().__init__(method)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit feature selector for time series data

        Args:
            X: Input features (3D for sequence models: samples, timesteps, features)
            y: Target values
            feature_names: Feature names
        """
        # For sequence data (3D), flatten the time dimension
        if len(X.shape) == 3:
            samples, timesteps, features = X.shape
            X_flat = X.reshape(samples, timesteps * features)

            # Generate feature names if not provided
            if feature_names is None:
                feature_names = []
                for t in range(timesteps):
                    for f in range(features):
                        feature_names.append(f"t-{timesteps-t}_feature_{f}")

            # Call parent fit method with flattened data
            super().fit(X_flat, y, feature_names)

            # Store original shape for transform
            self.original_shape = (samples, timesteps, features)
        else:
            # For non-sequence data, use parent fit method
            super().fit(X, y, feature_names)
            self.original_shape = None

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform time series data

        Args:
            X: Input features

        Returns:
            Transformed data
        """
        # For sequence data (3D), flatten, transform, and reshape back
        if len(X.shape) == 3 and self.original_shape is not None:
            samples, timesteps, features = X.shape
            X_flat = X.reshape(samples, timesteps * features)

            # Transform flattened data
            X_transformed = super().transform(X_flat)

            # For PCA, we can't reshape back to 3D
            if self.method == 'pca':
                return X_transformed

            # For other methods, reshape back to 3D
            # This assumes all features at each timestep are either selected or not
            new_features = X_transformed.shape[1] // timesteps
            return X_transformed.reshape(samples, timesteps, new_features)
        else:
            # For non-sequence data, use parent transform method
            return super().transform(X)
