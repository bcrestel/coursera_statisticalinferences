# ppv.py
# Benjamin Crestel, 2020-10-09

from typing import Dict


def calculate_ppv(
    percentage_experiments_H1true: float, alpha_level: float, power: float, p_hacking_ratio: float = 0.0
) -> Dict[str, float]:
    """
    Calculate the Positive Predictive Value (PPV) of a p-value

    :param percentage_experiments_H1true: percentage of experiments for which the alternative hypothesis is true
    :param alpha_level: significance level
    :param power: power of the study
    :param p_hacking_ratio: percentage of negative results presented as positive results
    :return:
    """
    TP = percentage_experiments_H1true * power
    FN = percentage_experiments_H1true * (1.0 - power)
    TP += p_hacking_ratio * FN
    FN = (1.0 - p_hacking_ratio) * FN

    FP = (1.0 - percentage_experiments_H1true) * alpha_level
    TN = (1.0 - percentage_experiments_H1true) * (1.0 - alpha_level)
    FP += p_hacking_ratio * TN
    TN = (1.0 - p_hacking_ratio) * TN

    PPV = TP / (TP + FP)
    FPRP = FP / (TP + FP)
    return {"TP": TP, "FN": FN, "FP": FP, "TN": TN, "PPV": PPV, "FPRP": FPRP}
