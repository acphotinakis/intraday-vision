# src/utils/custom_stats.py

import numpy as np
from collections import namedtuple

# Define a LinregressResult namedtuple with extra intercept_stderr
LinregressResult = namedtuple(
    "LinregressResult",
    ["slope", "intercept", "rvalue", "pvalue", "stderr", "intercept_stderr"]
)


def linregress(x, y):
    """
    Simple linear regression calculation returning a LinregressResult.

    Parameters
    ----------
    x, y : array-like
        Arrays of independent and dependent variables.

    Returns
    -------
    LinregressResult
        Namedtuple with slope, intercept, rvalue, pvalue, stderr, intercept_stderr.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must be the same length")
    if x.size < 2:
        raise ValueError("At least 2 data points are required")

    n = len(x)
    xmean = np.mean(x)
    ymean = np.mean(y)

    ssxm, ssxym, _, ssym = np.cov(x, y, bias=True).flat

    # correlation coefficient
    rvalue = ssxym / np.sqrt(ssxm * ssym) if ssxm > 0 and ssym > 0 else 0.0
    slope = ssxym / ssxm
    intercept = ymean - slope * xmean

    # Standard errors
    df = n - 2
    stderr = np.sqrt((1 - rvalue**2) * ssym / ssxm / df)
    intercept_stderr = stderr * np.sqrt(np.mean(x**2))

    # Simple p-value using t-statistic
    t_stat = slope / stderr
    pvalue = 2 * (1 - _t_cdf(np.abs(t_stat), df))

    return LinregressResult(slope, intercept, rvalue, pvalue, stderr, intercept_stderr)


def _t_cdf(t, df):
    """CDF of Student's t using approximation for simplicity."""
    from scipy.stats import t as student_t
    return student_t.cdf(t, df)
