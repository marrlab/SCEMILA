import matplotlib.pyplot as plt
import math
import os
import numpy as np

CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def plot(sc_df, cols=16, dpi=150, show_coordinates=True, show_scalebar=False, path_save=None):
    '''Plot single cell images in a grid.'''

    # load image array
    im_path = os.path.join(os.path.dirname(
        sc_df.iloc[0].im_path), 'processed', 'stacked_images.npy')
    im_ar = np.load(im_path)

    scale_size = 15/cols
    rows = math.ceil(im_ar.shape[0]/cols)

    fig = plt.figure(constrained_layout=False,
                     figsize=(15, (rows)*scale_size), dpi=dpi)
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)

    counter = 0
    for gs_sub in gs:

        ax = plt.subplot(gs_sub)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_coordinates:
            coordinates = CHARS[counter % cols] + \
                str(math.ceil((counter+1)/cols))
            ax.text(5, 28, coordinates, alpha=0.4)

        if show_scalebar and counter == 0:
            # 10 mikrometer = 57,8px
            ax.plot((8, 8), (72 - 29, 72 + 29), c='k')
            ax.plot((5, 11), (72 - 29, 72 - 29), c='k')
            ax.plot((5, 11), (72 + 29, 72 + 29), c='k')

        image = im_ar[sc_df.iloc[counter]['im_id'], ...]
        ax.imshow(image)

        counter += 1

        if(counter >= im_ar.shape[0]):
            break

        if(counter >= len(sc_df)):
            break

    plt.show()

    if path_save is not None:
        fig.savefig(path_save, bbox_inches='tight')
