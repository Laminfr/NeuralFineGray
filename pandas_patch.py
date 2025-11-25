"""
Pandas Compatibility Patch for auton_survival and lifelines

The package `auton_survival` and `lifelines` depend on older versions of `pandas` 
and fail with modern Python environments (Python ≥3.10 and Pandas ≥2.0).

Several functions inside `lifelines` still rely on deprecated pandas methods such as:
- Series.iteritems() (removed in pandas 2.2+)
- DataFrame.describe(datetime_is_numeric=...) (removed in pandas 2.0+)

This patch:
1. Restores `Series.iteritems` by redirecting it to `.items`
2. Wraps DataFrame `.describe()` so unsupported arguments are dropped
3. Silences deprecated warnings that make the console unreadable

Usage:
    Instead of: import pandas as pd
    Use:        from pandas_patch import pd
"""

import pandas as pd
import warnings

# Suppress pandas warnings about deprecated features
warnings.filterwarnings('ignore', category=FutureWarning, module='lifelines')
warnings.filterwarnings('ignore', category=FutureWarning, module='auton_survival')

# Patch 1: Restore Series.iteritems (removed in pandas 2.2+)
if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items

# Patch 2: Fix DataFrame.describe() compatibility
_old_describe = pd.DataFrame.describe

def _describe_compat(self, *args, **kwargs):
    """
    Wrapper for DataFrame.describe() that removes unsupported arguments.
    
    The 'datetime_is_numeric' parameter was removed in pandas 2.0+
    but lifelines still tries to use it.
    """
    kwargs.pop("datetime_is_numeric", None)
    return _old_describe(self, *args, **kwargs)

pd.DataFrame.describe = _describe_compat

# Export patched pandas
__all__ = ['pd']
