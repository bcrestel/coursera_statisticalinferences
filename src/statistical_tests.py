# statistical_tests.py
# Benjamin Crestel, 2020-08-07

import numpy as np
from scipy.stats import t


def two_tailed_t_test(samples: np.ndarray, H0: float):
    """
    Calculate a two-tailed t-test on the samples
    null hypothesis: samples_mean = H0

    :param samples: array of shape (size of each sample, number of simulations)
    :param H0: value of the mean of the distribution under the null hypothesis
    :return:
        t_value (distribution t(df=N-1)); length 'number of simulations'
        p_value (2-tailed t-test); length 'number of simulations'
    """
    empirical_mean = np.mean(samples, axis=0)
    number_samples = samples.shape[0]
    standard_error = np.std(samples, ddof=1, axis=0) / np.sqrt(number_samples)
    t_value = (empirical_mean - H0) / standard_error
    p_value = 2.0 * (1.0 - t(df=number_samples - 1).cdf(np.abs(t_value)))
    return t_value, p_value


def likelihood_ratio(
    param: np.ndarray, likelihood: np.ndarray, param1: float, param2: float
) -> float:
    """
    Calculate likelihood ratio L(theta=param1, D=D_0) / L(theta=param2, D=D_0)

    :param param: array of values of parameters (parameter of the binomial)
    :param likelihood: likelihood values (corresponding to the parameters)
    :param param1: first parameter to test
    :param param2: second paramter to test
    :return: likelihood ratio
    """
    index1 = np.argmin(np.abs(param - param1))
    index2 = np.argmin(np.abs(param - param2))
    return likelihood[index1] / likelihood[index2]
