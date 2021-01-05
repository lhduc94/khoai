from .heatmap import heatmap
import pandas as pd


def plot_corr(df: pd.DataFrame, columns=None, **params):
    """
            A function plot correlation heatmap.
                    Parameters:
                                df: DataFrame
                                    True label
                                columns: List of columns
                                    If columns = None, get all columns of df
                                **params: The parameters of figure.
                                    figsize: Tuple of figure size. Default (17,10)
                                    n_colors: Number of color. Default 256
                                    size_scale: Scale point. Default 500

                    Returns:
                                figure: Figure
        """

    if columns is None:
        columns = df.columns
    corr = df[columns].corr()
    corr = pd.melt(corr.reset_index(),
                   id_vars='index')
    corr.columns = ['x', 'y', 'value']
    figure = heatmap(x=corr['x'], y=corr['y'], size=corr['value'].abs(), color=corr['value'], **params)

    return figure
