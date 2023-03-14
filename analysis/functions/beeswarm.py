import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def beeswarm_coordinates(
        df,
        val_col,
        xscale="log",
        figsize=(
            9,
            9),
    pointsize=7,
        prefix=''):
    '''small hack: don't show image, but use seaborn to calculate coordinates.
    Could not find a solution performing as well as the seaborn implementation'''
    df = df.sort_values(by=val_col, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale)
    fig = sns.swarmplot(data=df, x=val_col, size=pointsize)
    # get precise data coordinates
    x, y = np.array(ax.collections[0].get_offsets()).T
    df[prefix + 'x'] = x
    df[prefix + 'y'] = y
    xlim = min(df[prefix + 'x'])*0.95, max(df[prefix + 'x'])*1.05
    ylim = min(df[prefix + 'y'])*0.95, max(df[prefix + 'y'])*0.95
    plt.close('all')
    return df, xlim, ylim
