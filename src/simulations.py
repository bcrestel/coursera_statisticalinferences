# simulations.py
# Benjamin Crestel, 2020-08-07

import numpy as np


def simulate_normal(
    sample_mean: float, sample_std: float, sample_size: int, number_simulations: int
):
    """
    Sample iid, normally distributed data

    :param sample_mean: mean of the distribution
    :param sample_std: standard deviation of the distribution
    :param sample_size: number of samples per simulation
    :param number_simulations: number of simulations
    :return: array of shape (sample_size, number_simulations)
    """
    return (
        sample_mean + sample_std * np.random.randn(sample_size * number_simulations)
    ).reshape((sample_size, number_simulations))


def simulate_normal_array(
    sample_mean: np.ndarray, sample_std: np.ndarray, number_simulations: int
):
    """
    Sample iid, normally distributed data

    :param sample_mean: mean of the distribution; length = number of samples
    :param sample_std: standard deviation of the distribution; length = number of samples
    :param number_simulations: number of simulations
    :return: array of shape (sample_size, number_simulations)
    """
    sample_size = len(sample_mean)
    if sample_size != len(sample_std):
        raise ValueError(
            "sample_mean and sample_std must have the same length; "
            + f"got {sample_size} and {len(sample_std)}"
        )

    std = (
        np.random.randn(sample_size * number_simulations).reshape(
            (number_simulations, sample_size)
        )
        * sample_std
    )
    return (std + sample_mean).T
