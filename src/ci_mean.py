# ci_mean.py
# Benjamin Crestel, 2020-12-16
# Copyright (c) 2020 Element AI. All rights reserved.

import numpy as np
from scipy.stats import t

from simulations import simulate_normal

class CIMean():

    def run(self, mean:float, std:float, sample_size: int, number_simulations: int):
        data = simulate_normal(sample_mean=mean, sample_std=std, sample_size=sample_size, number_simulations=number_simulations)

        # Number of CI containing the data:
        means = data.mean(axis=0)
        std = data.std(axis=0)
        se = std / np.sqrt(sample_size)
        deviation = t.ppf(0.975, df=sample_size-1)
        ci_low = means - deviation * se
        ci_high = means + deviation * se
        percentage_ci = sum((ci_low < mean) * (mean < ci_high)) / number_simulations

        # Capture percentage
        capture_percentage = sum((ci_low[0] < means[1:]) * (means[1:] < ci_high[0])) / number_simulations

        return percentage_ci, capture_percentage, [ci_low[0], ci_high[0]]
