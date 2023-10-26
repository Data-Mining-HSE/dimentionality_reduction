import pandas as pd


def interquartile_range(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    The approach of outliers reduction uses the Interquartile Range methodology

    References:
    https://www.sciencedirect.com/science/article/pii/S2772662223000048?via%3Dihub
    """
    quantile_1 = data.quantile(0.25)
    quantile_3 = data.quantile(0.75)

    IQR = quantile_3 - quantile_1

    left_bound = quantile_1 - 1.5 * IQR
    right_bound = quantile_3 + 1.5 * IQR

    outliers_mask = (data < left_bound) | (data > right_bound)
    not_outliers = data[~outliers_mask]
    outliers_dropped = not_outliers.dropna().reset_index(drop=True)
    if verbose:
        heading = 'The outlier data detected'
        print(heading, '\n', '-'*len(heading))
        print(outliers_mask.sum().to_string())
        print('-'*len(heading), '\n', f'Total deleted rows: {len(data) - len(outliers_dropped)}')
    return outliers_dropped
