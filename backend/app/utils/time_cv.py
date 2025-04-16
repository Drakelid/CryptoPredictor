import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Generator, Callable
import logging
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

class TimeSeriesCV:
    """Enhanced time series cross-validation with expanding window option"""
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        expanding_window: bool = True
    ):
        """
        Initialize time series cross-validation
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (optional)
            gap: Gap between train and test sets
            max_train_size: Maximum size of training set (optional)
            expanding_window: Whether to use expanding window (True) or sliding window (False)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        self.expanding_window = expanding_window
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices
        
        Args:
            X: Input features
            y: Target values (optional, not used)
        
        Yields:
            Train and test indices for each split
        """
        n_samples = len(X)
        
        # Calculate test size if not provided
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Calculate indices
        indices = np.arange(n_samples)
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test start and end
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # Ensure test set is within bounds
            if test_start < 0:
                break
            
            # Calculate train end (considering gap)
            train_end = test_start - self.gap
            
            # Calculate train start
            if self.expanding_window:
                # Expanding window: always start from the beginning
                train_start = 0
            else:
                # Sliding window: use max_train_size if provided
                if self.max_train_size is not None:
                    train_start = max(0, train_end - self.max_train_size)
                else:
                    # Default sliding window: same size as test set
                    train_start = max(0, train_end - test_size)
            
            # Ensure train set is not empty
            if train_end <= train_start:
                break
            
            # Get train and test indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, y: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """
        Get number of splits
        
        Args:
            X: Input features (optional)
            y: Target values (optional)
        
        Returns:
            Number of splits
        """
        return self.n_splits
    
    def plot_splits(self, X: Union[np.ndarray, pd.DataFrame], save_path: Optional[str] = None):
        """
        Plot cross-validation splits
        
        Args:
            X: Input features
            save_path: Path to save the plot (optional)
        """
        n_samples = len(X)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Generate splits
        for i, (train_indices, test_indices) in enumerate(self.split(X)):
            # Plot train set
            plt.plot(
                train_indices,
                [i] * len(train_indices),
                'o-',
                color='blue',
                label='Train set' if i == 0 else None
            )
            
            # Plot gap
            if self.gap > 0:
                gap_start = train_indices[-1] + 1
                gap_end = test_indices[0]
                plt.plot(
                    range(gap_start, gap_end),
                    [i] * (gap_end - gap_start),
                    'o-',
                    color='gray',
                    label='Gap' if i == 0 else None
                )
            
            # Plot test set
            plt.plot(
                test_indices,
                [i] * len(test_indices),
                'o-',
                color='red',
                label='Test set' if i == 0 else None
            )
        
        # Set labels and title
        plt.xlabel('Sample index')
        plt.ylabel('CV iteration')
        plt.title(f"{'Expanding' if self.expanding_window else 'Sliding'} Window Time Series CV")
        plt.legend()
        
        # Set y-axis limits
        plt.ylim(-0.5, self.n_splits - 0.5)
        
        # Set x-axis limits
        plt.xlim(-0.5, n_samples - 0.5)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class WalkForwardCV:
    """Walk-forward cross-validation for time series"""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.7,
        test_size: Optional[int] = None,
        gap: int = 0,
        purge_overlap: bool = True
    ):
        """
        Initialize walk-forward cross-validation
        
        Args:
            n_splits: Number of splits
            train_size: Proportion of data to use for training
            test_size: Size of test set (optional)
            gap: Gap between train and test sets
            purge_overlap: Whether to purge overlapping samples
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.purge_overlap = purge_overlap
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices
        
        Args:
            X: Input features
            y: Target values (optional, not used)
        
        Yields:
            Train and test indices for each split
        """
        n_samples = len(X)
        
        # Calculate test size if not provided
        if self.test_size is None:
            test_size = int(n_samples * (1 - self.train_size) / self.n_splits)
        else:
            test_size = self.test_size
        
        # Calculate indices
        indices = np.arange(n_samples)
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test start and end
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # Ensure test set is within bounds
            if test_start < 0:
                break
            
            # Calculate train end (considering gap)
            train_end = test_start - self.gap
            
            # Calculate train start
            train_size = int(n_samples * self.train_size)
            train_start = max(0, train_end - train_size)
            
            # Ensure train set is not empty
            if train_end <= train_start:
                break
            
            # Get train and test indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            # Purge overlapping samples if needed
            if self.purge_overlap and i > 0:
                # Get previous test indices
                prev_test_start = test_start + test_size
                prev_test_end = test_end + test_size
                prev_test_indices = indices[prev_test_start:prev_test_end]
                
                # Remove overlapping samples from train set
                train_indices = np.setdiff1d(train_indices, prev_test_indices)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, y: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """
        Get number of splits
        
        Args:
            X: Input features (optional)
            y: Target values (optional)
        
        Returns:
            Number of splits
        """
        return self.n_splits
    
    def plot_splits(self, X: Union[np.ndarray, pd.DataFrame], save_path: Optional[str] = None):
        """
        Plot cross-validation splits
        
        Args:
            X: Input features
            save_path: Path to save the plot (optional)
        """
        n_samples = len(X)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Generate splits
        for i, (train_indices, test_indices) in enumerate(self.split(X)):
            # Plot train set
            plt.plot(
                train_indices,
                [i] * len(train_indices),
                'o-',
                color='blue',
                label='Train set' if i == 0 else None
            )
            
            # Plot gap
            if self.gap > 0:
                gap_start = train_indices[-1] + 1
                gap_end = test_indices[0]
                plt.plot(
                    range(gap_start, gap_end),
                    [i] * (gap_end - gap_start),
                    'o-',
                    color='gray',
                    label='Gap' if i == 0 else None
                )
            
            # Plot test set
            plt.plot(
                test_indices,
                [i] * len(test_indices),
                'o-',
                color='red',
                label='Test set' if i == 0 else None
            )
        
        # Set labels and title
        plt.xlabel('Sample index')
        plt.ylabel('CV iteration')
        plt.title(f"Walk-Forward CV (Purge: {self.purge_overlap})")
        plt.legend()
        
        # Set y-axis limits
        plt.ylim(-0.5, self.n_splits - 0.5)
        
        # Set x-axis limits
        plt.xlim(-0.5, n_samples - 0.5)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class CombinedTimeSeriesCV:
    """Combined time series cross-validation with multiple methods"""
    
    def __init__(
        self,
        methods: List[str] = ['expanding', 'sliding', 'walk_forward'],
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize combined time series cross-validation
        
        Args:
            methods: List of CV methods to use
            n_splits: Number of splits
            test_size: Size of test set (optional)
            gap: Gap between train and test sets
        """
        self.methods = methods
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
        # Create CV objects
        self.cv_objects = {}
        
        if 'expanding' in methods:
            self.cv_objects['expanding'] = TimeSeriesCV(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap,
                expanding_window=True
            )
        
        if 'sliding' in methods:
            self.cv_objects['sliding'] = TimeSeriesCV(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap,
                expanding_window=False
            )
        
        if 'walk_forward' in methods:
            self.cv_objects['walk_forward'] = WalkForwardCV(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap
            )
        
        if 'sklearn' in methods:
            self.cv_objects['sklearn'] = TimeSeriesSplit(
                n_splits=n_splits,
                gap=gap,
                test_size=test_size
            )
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_factory: Callable[[], Any],
        scoring_function: Callable[[np.ndarray, np.ndarray], float]
    ) -> Dict[str, List[float]]:
        """
        Evaluate model using multiple CV methods
        
        Args:
            X: Input features
            y: Target values
            model_factory: Function that returns a new model instance
            scoring_function: Function to score predictions
        
        Returns:
            Dictionary mapping CV methods to lists of scores
        """
        results = {}
        
        for method, cv in self.cv_objects.items():
            logger.info(f"Evaluating with {method} CV")
            
            scores = []
            
            for train_indices, test_indices in cv.split(X):
                # Get train/test data
                X_train = X[train_indices] if isinstance(X, np.ndarray) else X.iloc[train_indices]
                y_train = y[train_indices] if isinstance(y, np.ndarray) else y.iloc[train_indices]
                X_test = X[test_indices] if isinstance(X, np.ndarray) else X.iloc[test_indices]
                y_test = y[test_indices] if isinstance(y, np.ndarray) else y.iloc[test_indices]
                
                # Create and train model
                model = model_factory()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate score
                score = scoring_function(y_test, y_pred)
                scores.append(score)
            
            results[method] = scores
        
        return results
    
    def plot_results(self, results: Dict[str, List[float]], metric_name: str = 'Score', save_path: Optional[str] = None):
        """
        Plot CV results
        
        Args:
            results: Dictionary mapping CV methods to lists of scores
            metric_name: Name of the metric
            save_path: Path to save the plot (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot results
        for method, scores in results.items():
            plt.plot(range(1, len(scores) + 1), scores, 'o-', label=method)
        
        # Calculate mean scores
        mean_scores = {method: np.mean(scores) for method, scores in results.items()}
        
        # Add mean lines
        for method, mean_score in mean_scores.items():
            plt.axhline(y=mean_score, linestyle='--', color='gray', alpha=0.5)
            plt.text(
                len(list(results.values())[0]) + 0.1,
                mean_score,
                f"{method}: {mean_score:.4f}",
                verticalalignment='center'
            )
        
        # Set labels and title
        plt.xlabel('CV Fold')
        plt.ylabel(metric_name)
        plt.title(f"Cross-Validation Results ({metric_name})")
        plt.legend()
        
        # Set x-axis limits
        plt.xlim(0.5, len(list(results.values())[0]) + 0.5)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
