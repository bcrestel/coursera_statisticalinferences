# plots.py
# Benjamin Crestel, 2020-08-07

import numpy as np
from scipy.stats import binom


def likelihood_curve_binomial(
    nb_tries: int, nb_success: int, ax, normalize: bool = True, label=""
):
    """
    Plot the likelihood curve for a binomial distribution

    :param nb_tries: Number of trials of the experiment (n)
    :param nb_success: Number of successes in these n trials
    :param ax: matplotlib axes
    :param normalize: normalize the curves or not
    :param label: label for the plot
    :return:
    """
    proba_success = np.linspace(0.0, 1.0, 101)
    likelihood = binom(n=nb_tries, p=proba_success).pmf(nb_success)
    if normalize:
        likelihood = likelihood / np.max(likelihood)
    ax.plot(proba_success, likelihood, label=label)
    ax.set_xlabel("Probablility of success $p$ for that coin")
    ax.set_ylabel("Likelihood")
    ax.set_title(f"Likelihood Curve for {nb_success} sucesses out of {nb_tries} tries")
    return proba_success, likelihood
