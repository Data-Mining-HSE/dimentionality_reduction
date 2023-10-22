import pandas as pd


def interquartile_range(data: pd.DataFrame) -> pd.DataFrame:
    quantile_1 = data.quantile(0.25)
    quantile_3 = data.quantile(0.75)

    IQR = quantile_3 - quantile_1

    left_bound = quantile_1 - 1.5 * IQR
    right_bound = quantile_3 + 1.5 * IQR

    outliers_mask = (data < left_bound) | (data > right_bound)
    not_outliers = data[~outliers_mask]
    outliers_dropped = not_outliers.dropna().reset_index()
    return outliers_dropped
