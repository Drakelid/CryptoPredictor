"""
Custom pandas initialization module to fix deprecation warnings.
This module should be imported before any other imports that use pandas.
"""
import pandas as pd
import functools

# Store the original pct_change method
_original_pct_change = pd.Series.pct_change

# Create a wrapper function that sets fill_method=None by default
@functools.wraps(_original_pct_change)
def _patched_pct_change(self, periods=1, fill_method=None, limit=None, freq=None, **kwargs):
    """
    Patched version of pandas Series.pct_change that uses fill_method=None by default.
    This avoids the deprecation warning about fill_method='pad'.
    """
    return _original_pct_change(self, periods=periods, fill_method=fill_method, limit=limit, freq=freq, **kwargs)

# Replace the original method with our patched version
pd.Series.pct_change = _patched_pct_change

# Also patch DataFrame.pct_change
_original_df_pct_change = pd.DataFrame.pct_change

@functools.wraps(_original_df_pct_change)
def _patched_df_pct_change(self, periods=1, fill_method=None, limit=None, freq=None, **kwargs):
    """
    Patched version of pandas DataFrame.pct_change that uses fill_method=None by default.
    This avoids the deprecation warning about fill_method='pad'.
    """
    return _original_df_pct_change(self, periods=periods, fill_method=fill_method, limit=limit, freq=freq, **kwargs)

pd.DataFrame.pct_change = _patched_df_pct_change
