"""Utility to patch pandas to silence deprecation warnings.

If pandas is not installed, minimal stubs are used so the rest of the
application can still load. The full functionality that relies on pandas
will of course be unavailable in that case.
"""

import functools

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
    if 'DataFrame' not in pd.__dict__:
        raise ImportError
except Exception:  # pragma: no cover - fallback when pandas missing
    def _pct_change(self, *a, **k):
        return None

    pd = type('pandas', (), {
        'Series': type('Series', (), {'pct_change': _pct_change}),
        'DataFrame': type('DataFrame', (), {'pct_change': _pct_change})
    })

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
