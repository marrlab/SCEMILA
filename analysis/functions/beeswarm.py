import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def beeswarm_coordinates(df, val_col, xscale="log", figsize=(9, 9), pointsize=7):
    '''Swarmplot coordinate calculation is not available in Matplotlib to my knowledge.
    This workaround uses the implemented and efficient Seaborn function, to simply calculate
    the coordinates.'''
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale)
    fig = sns.swarmplot(data=df, x=val_col, size=pointsize)
    # get precise data coordinates
    x, y = np.array(ax.collections[0].get_offsets()).T
    df = df.sort_values(by=val_col, ascending=True)
    df['x'] = x
    df['y'] = y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close('all')
    return df, xlim, ylim
