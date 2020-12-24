# gpower.py
# Benjamin Crestel, 2020-12-23

from typing import Tuple

import numpy as np
from scipy.stats import nct, t


class GPower:
    def run_twogroup_indep_ttest(
        self, mean_diff: float, std: float, alpha_level: float, target_power: float
    ) -> int:
        """
        Power analysis for two-group independent t-test

        :param mean_diff: difference between the means of both population
        :param std: standard deviation of the population distribution
        :param alpha_level: alpha level for the t-test
        :param target_power: target power
        :return: sample size to reach target_power at the alpha_level for given mean_diff
        """
        init_sample_size = 100
        init_power = self.power_twogroup_indep_ttest(
            mean_diff=mean_diff,
            sample_size=init_sample_size,
            std=std,
            alpha_level=alpha_level,
        )
        if init_power > target_power:
            sample_size_bounds = (2, init_sample_size)
        else:
            sample_size_bounds = (init_sample_size, 1000000)

        tolerance = 0.01
        power_low, power_up = self.powers_twogroup_indep_ttest(
            mean_diff=mean_diff,
            sample_size_bounds=sample_size_bounds,
            std=std,
            alpha_level=alpha_level,
        )
        while (
            np.abs(power_low - power_up) > tolerance
            and np.diff(sample_size_bounds).item() > 1.0
        ):
            sample_size_mid = self._get_mid_sample_size(
                sample_size_bounds=sample_size_bounds
            )
            power_mid = self.power_twogroup_indep_ttest(
                mean_diff=mean_diff,
                sample_size=sample_size_mid,
                std=std,
                alpha_level=alpha_level,
            )
            if power_mid < target_power:
                sample_size_bounds = (sample_size_mid, sample_size_bounds[1])
            else:
                sample_size_bounds = (sample_size_bounds[0], sample_size_mid)

        return sample_size_bounds

    @staticmethod
    def _get_mid_sample_size(sample_size_bounds: Tuple[int]):
        return int(np.mean(sample_size_bounds))

    def powers_twogroup_indep_ttest(
        self,
        mean_diff: float,
        sample_size_bounds: Tuple[int],
        std: float,
        alpha_level: float,
    ):
        power_low = self.power_twogroup_indep_ttest(
            mean_diff=mean_diff,
            sample_size=sample_size_bounds[0],
            std=std,
            alpha_level=alpha_level,
        )
        power_up = self.power_twogroup_indep_ttest(
            mean_diff=mean_diff,
            sample_size=sample_size_bounds[1],
            std=std,
            alpha_level=alpha_level,
        )
        return power_low, power_up

    @staticmethod
    def power_twogroup_indep_ttest(
        mean_diff: float, sample_size: int, std: float, alpha_level: float
    ) -> float:
        """
        Compute power of two-group independent t-test (2 tails)
        Ref: https://www.real-statistics.com/students-t-distribution/statistical-power-of-the-t-tests/
        https://en.wikipedia.org/wiki/Noncentral_t-distribution

        :param mean_diff: difference between the means of both population
        :param sample_size: number of samples in each population
        :param std: standard deviation of the population distribution
        :param alpha_level: alpha level for the t-test
        :return: power of the test
        """
        df = 2 * sample_size - 2
        cohen_effect_size = mean_diff / std
        delta = cohen_effect_size * np.sqrt(sample_size * 0.5)
        t_critical = t(df).ppf(1.0 - 0.5 * alpha_level)

        distr = nct(df=df, nc=delta)
        beta = distr.cdf(t_critical) - distr.cdf(-t_critical)
        power = 1.0 - beta
        return power

    def run_twogroup_indep_ttest2(
            self, mean_diff: float, cohen_effect_size: float, alpha_level: float, target_power: float
    ) -> int:
        """
        Power analysis for two-group independent t-test

        :param mean_diff: difference between the means of both population
        :param cohen_effect_size:
        :param alpha_level: alpha level for the t-test
        :param target_power: target power
        :return: sample size to reach target_power at the alpha_level for given mean_diff
        """
        init_sample_size = 100
        init_power = self.power_twogroup_indep_ttest2(
            mean_diff=mean_diff,
            sample_size=init_sample_size,
            cohen_effect_size=cohen_effect_size,
            alpha_level=alpha_level,
        )
        if init_power > target_power:
            sample_size_bounds = (2, init_sample_size)
        else:
            sample_size_bounds = (init_sample_size, 1000000)

        tolerance = 0.01
        power_low, power_up = self.powers_twogroup_indep_ttest2(
            mean_diff=mean_diff,
            sample_size_bounds=sample_size_bounds,
            cohen_effect_size=cohen_effect_size,
            alpha_level=alpha_level,
        )
        while (
                np.abs(power_low - power_up) > tolerance
                and np.diff(sample_size_bounds).item() > 1.0
        ):
            sample_size_mid = self._get_mid_sample_size(
                sample_size_bounds=sample_size_bounds
            )
            power_mid = self.power_twogroup_indep_ttest2(
                mean_diff=mean_diff,
                sample_size=sample_size_mid,
                cohen_effect_size=cohen_effect_size,
                alpha_level=alpha_level,
            )
            if power_mid < target_power:
                sample_size_bounds = (sample_size_mid, sample_size_bounds[1])
            else:
                sample_size_bounds = (sample_size_bounds[0], sample_size_mid)

        return sample_size_bounds

    @staticmethod
    def power_twogroup_indep_ttest2(
            mean_diff: float, sample_size: int, cohen_effect_size: float, alpha_level: float
    ) -> float:
        """
        Compute power of two-group independent t-test (2 tails)
        Ref: https://www.real-statistics.com/students-t-distribution/statistical-power-of-the-t-tests/
        https://en.wikipedia.org/wiki/Noncentral_t-distribution

        :param mean_diff: difference between the means of both population
        :param sample_size: number of samples in each population
        :param cohen_effect_size:
        :param alpha_level: alpha level for the t-test
        :return: power of the test
        """
        df = 2 * sample_size - 2
        delta = cohen_effect_size * np.sqrt(sample_size * 0.5)
        t_critical = t(df).ppf(1.0 - 0.5 * alpha_level)

        distr = nct(df=df, nc=delta)
        beta = distr.cdf(t_critical) - distr.cdf(-t_critical)
        power = 1.0 - beta
        return power

    def powers_twogroup_indep_ttest2(
            self,
            mean_diff: float,
            sample_size_bounds: Tuple[int],
            cohen_effect_size: float,
            alpha_level: float,
    ):
        power_low = self.power_twogroup_indep_ttest2(
            mean_diff=mean_diff,
            sample_size=sample_size_bounds[0],
            cohen_effect_size=cohen_effect_size,
            alpha_level=alpha_level,
        )
        power_up = self.power_twogroup_indep_ttest2(
            mean_diff=mean_diff,
            sample_size=sample_size_bounds[1],
            cohen_effect_size=cohen_effect_size,
            alpha_level=alpha_level,
        )
        return power_low, power_up

