
import pandas as pd
from matplotlib import pyplot as plt


def get_histograms(dataset: pd.DataFrame) -> None:
    dataset.hist(figsize=(9, 7))


def get_boxplot(dataset: pd.DataFrame) -> None:
    figure, axes = plt.subplots(ncols=len(dataset.columns), nrows=1)

    figure.set_figheight(3)
    figure.set_figwidth(8)

    for column_id, column in enumerate(dataset.columns):
        pd.DataFrame(dataset[column]).boxplot(ax=axes[column_id])

    figure.tight_layout()
 
