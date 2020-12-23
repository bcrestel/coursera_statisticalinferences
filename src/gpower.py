# gpower.py
# Benjamin Crestel, 2020-12-23

import numpy as np
from scipy.stats import nct, t


def gpower(mean_diff: float, alpha_level: float):
    """
    Power analysis for two-group independent t-test

    :param mean_diff: difference between the means of both population
    :param alpha_level:
    :return:
    """
    return None


def power_twogroup_indep_ttest(
    mean_diff: float, sample_size: int, std: float, alpha_level: float
):
    """
    Compute power of two-group independent t-test
    Ref: https://www.real-statistics.com/students-t-distribution/statistical-power-of-the-t-tests/

    :param mean_diff: difference between the means of both population
    :param sample_size: number of samples in each population
    :param std: standard deviation
    :param alpha_level:
    :return:
    """
    df = 2 * sample_size - 2
    cohen_effect_size = mean_diff / std
    delta = cohen_effect_size * np.sqrt(sample_size * 0.5)
    t_critical = t(df).ppf(1.0 - 0.5 * alpha_level)

    distr = nct(df=df, nc=delta)
    beta = distr.cdf(t_critical) - distr.cdf(-t_critical)
    power = 1.0 - beta
    return power
