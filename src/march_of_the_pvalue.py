# march_of_the_pvalue.py
# Benjamin Crestel, 2020-12-23

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from simulations import simulate_normal


def march_of_the_pvalue(number_simulations: int, mean_diff: float, sample_size: int):
    """
    Simulate repeated t-test on 2 samples

    :param number_simulations: number of simulations to run
    :param mean_diff: difference between the means of both population
    :param sample_size: number of samples in each population
    :return:
    """
    # Run all t-tests
    ref_mean = 100.0
    ref_std = 15.0
    samples1 = simulate_normal(
        sample_mean=ref_mean,
        sample_std=ref_std,
        sample_size=sample_size,
        number_simulations=number_simulations,
    )
    samples2 = simulate_normal(
        sample_mean=ref_mean + mean_diff,
        sample_std=ref_std,
        sample_size=sample_size,
        number_simulations=number_simulations,
    )
    ttest = ttest_ind(samples1, samples2)
    pvalues = ttest.pvalue

    # Plot histogram of p-values
    fig, ax = plt.subplots(1, 1)
    ax.hist(pvalues, range=(0, 1), bins=20, density=True)

    return pvalues, ax
