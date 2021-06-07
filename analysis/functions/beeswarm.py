import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def beeswarm_coordinates(df, val_col, xscale="log", figsize=(9,9), pointsize=7):
    '''small hack: don't show image, but use seaborn to calculate coordinates.
    Could not find a solution performing as well as the seaborn implementation'''
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