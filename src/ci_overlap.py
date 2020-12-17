# ci_overlap.py
# Benjamin Crestel, 2020-12-16
# Copyright (c) 2020 Element AI. All rights reserved.

from typing import Tuple

import numpy as np
from scipy.stats import norm, t

from simulations import simulate_normal


class CIOverlap:
    @staticmethod
    def generate_data(
        means: Tuple[float, float], std: Tuple[float, float], sample_size: int
    ):
        data = []
        for mean, std in zip(means, std):
            data.append(
                simulate_normal(
                    sample_mean=mean,
                    sample_std=std,
                    sample_size=sample_size,
                    number_simulations=1,
                ).flatten()
            )

        data = np.array(data)
        return data

    @staticmethod
    def calculate_ci(datas: np.ndarray, percentage_confidence: float):
        """

        :param datas: input datas; shape (2, sample_size)
        :return:
        """
        mean = datas.mean(axis=1)
        nb_samples = datas.shape[1]
        std = datas.std(axis=1)
        se = std / np.sqrt(nb_samples)

        typeIerr = 1.0 - percentage_confidence
        quantile_high = 1.0 - typeIerr * 0.5
        deviation_high = norm.ppf(quantile_high)
        quantile_low = typeIerr * 0.5
        deviation_low = norm.ppf(quantile_low)
        # print(quantile_low, quantile_high)

        ci_high = mean + deviation_high * se
        ci_low = mean + deviation_low * se

        CIs = []
        for _ci_low, _ci_high in zip(ci_low, ci_high):
            CIs.append([_ci_low, _ci_high])

        # CI for the difference
        mean_diff = mean[1] - mean[0]
        se_diff = np.sqrt(std[0] ** 2 / nb_samples + std[1] ** 2 / nb_samples)
        deviation_diff_high = t.ppf(quantile_high, df=2 * nb_samples - 2)
        deviation_diff_low = t.ppf(quantile_low, df=2 * nb_samples - 2)
        CIs.append(
            [
                mean_diff + deviation_diff_low * se_diff,
                mean_diff + deviation_diff_high * se_diff,
            ]
        )
        # print(deviation_diff_low, deviation_diff_high)

        return CIs

    def run(
        self, means: Tuple[float, float], std: Tuple[float, float], sample_size: int
    ):
        self.means = means
        self.std = std
        self.sample_size = sample_size

        self.data = self.generate_data(means=means, std=std, sample_size=sample_size)
        self.CIs = self.calculate_ci(datas=self.data, percentage_confidence=0.95)
