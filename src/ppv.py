# ppv.py
# Benjamin Crestel, 2020-10-09

def calculate_ppv(percentage_experiments_H0true: float,
        alpha_level: float,
        power: float
        ):
    """
    Calculate the Positive Predictive Value (PPV) of a p-value

    :param percentage_experiments_H0true: percentage of experiments for which the null is true
    :param alpha_level: significance level
    :param power: power of the study
    :return:
    """
    TP = (1.0-percentage_experiments_H0true) * power
    FN = (1.0-percentage_experiments_H0true) * (1.0 - power)
    FP = percentage_experiments_H0true * alpha_level
    TN = percentage_experiments_H0true * (1.0 - alpha_level)
    PPV = TP / (TP + FP)
    FPRP = FP / (TP + FP)
    return {'TP':TP, 'FN':FN, 'FP':FP, 'TN':TN, 'PPV':PPV, 'FPRP':FPRP}

