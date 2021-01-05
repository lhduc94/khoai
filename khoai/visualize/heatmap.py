import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def heatmap(x, y, size, color, **params):
    n_colors = params.get('n_colors', 256)
    palette = sns.diverging_palette(20, 220, n=n_colors)
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    figure = plt.figure(figsize=params.get('figsize', None))
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
    ax = plt.subplot(plot_grid[:, :-1])

    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * params.get('size_scale', 500),
        c=color.map(value_to_color),
        marker=params.get('marker', 's')
    )

    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=params.get('x_rotation', 45), horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    ax = plt.subplot(plot_grid[:, -1])

    col_x = [0] * len(palette)
    bar_y = np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax.yaxis.tick_right()
    return figure
