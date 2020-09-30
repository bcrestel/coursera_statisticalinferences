# bayesian.py
# Benjamin Crestel, 2020-09-29

from scipy.stats import beta


def BayesFactorBinomial(
    number_flips: int,
    number_heads: int,
    prior_a: float,
    prior_b: float,
    point_null: float,
) -> float:
    """
    Calculate the Bayes Factor at point_null for a coin flip experiment with Beta prior

    :param number_flips: number of flipds
    :param number_heads: number of heads
    :param prior_a: alpha value of prior distribution
    :param prior_b: beta value of prior distribution
    :param point_null: value of the parameter where to calculate the Bayes Factors

    :return: Bayes factor
    """
    prior = beta.pdf(point_null, prior_a, prior_b)
    posterior = beta.pdf(
        point_null, prior_a + number_heads, prior_b + number_flips - number_heads
    )

    return posterior / prior
