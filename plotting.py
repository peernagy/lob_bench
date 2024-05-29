from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from partitioning import flatten


def hist(
        real,
        gen,
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
        *,
        bins: Optional[int|str|list[float]] = 'auto',
        binwidth: Optional[float]= None,
    ):
    """
    """
    real = pd.DataFrame(flatten(real))
    real['type'] = 'real'
    gen = pd.DataFrame(flatten(gen))
    gen['type'] = 'generated'
    data = pd.concat([real, gen], axis=0)
    data.columns = ['x', 'type']
    # display(data)
    sns.histplot(
        data, x='x', hue='type',
        alpha=0.5, stat='density', multiple='layer',
        common_bins=True, common_norm=False, binwidth=binwidth, bins=bins
    )
    # sns.histplot(gen, color='red', alpha=0.5, label='Generated', ax=plt.gca())
    # plt.legend()
    _finish_plot(title, xlabel, ylabel)
    if not isinstance(bins, str):
        print(f'bins: {bins}')

def scatter(
        score_df: pd.DataFrame,
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
    ):
    """
    """
    sns.scatterplot(data=score_df, x='group', y='score', hue='type')
    _finish_plot(title, xlabel, ylabel)

def line(
        score_df: pd.DataFrame,
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
    ):
    """
    """
    sns.lineplot(
        data=score_df, x='group', y='score', hue='type',
        errorbar=("pi", 0.95)
    )
    _finish_plot(title, xlabel, ylabel)

def _finish_plot(
        title: str= '',
        xlabel: str= 'Group',
        ylabel: str= 'Score',
    ):
    """
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
