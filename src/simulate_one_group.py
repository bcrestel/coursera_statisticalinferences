# simulate_one_group.py
# Benjamin Crestel, 2020-12-22

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from simulations import simulate_normal


def simulate_one_group(number_samples: int, mean: float = 100.0, std: float = 15.0):
    """
    Generate samples of IQ tests and plot

    :param number_samples: number of samples
    :param mean: mean of the distribution
    :param std: standard deviation of the distribution
    :return: samples and axes of the plot
    """
    samples = simulate_normal(
        sample_mean=100, sample_std=15, sample_size=number_samples, number_simulations=1
    )

    samples_mean = samples.mean()
    samples_std = samples.std()

    fig, ax = plt.subplots(1, 1)

    bounds = samples.min() - 10.0, samples.max() + 10.0
    ax.hist(samples, range=bounds)

    plt.text(100, 0.15 * number_samples, f"mean = {samples_mean:.2f}")
    plt.text(100, 0.1 * number_samples, f"std = {samples_std:.2f}")

    ax2 = ax.twinx()
    xx = np.linspace(bounds[0], bounds[1], 100)
    ax2.plot(xx, norm(100, 15).pdf(xx), "--r")

    return samples, (ax, ax2)
