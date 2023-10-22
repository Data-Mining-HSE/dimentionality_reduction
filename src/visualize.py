import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def get_subplots(ncols: int, figsize: tuple[int, int]) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(ncols=ncols, nrows=1)

    figure.set_figheight(figsize[0])
    figure.set_figwidth(figsize[1])

    figure.tight_layout()
    return figure, axes


def get_histograms(dataset: pd.DataFrame) -> None:
    _, axes = get_subplots(len(dataset.columns), figsize=(3, 10))

    for column_id, column in enumerate(dataset.columns):
        pd.DataFrame(dataset[column]).hist(ax=axes[column_id])


def get_boxplot(dataset: pd.DataFrame) -> None:
    _, axes = get_subplots(len(dataset.columns), figsize=(5, 9))

    for column_id, column in enumerate(dataset.columns):
        pd.DataFrame(dataset[column]).boxplot(ax=axes[column_id])
