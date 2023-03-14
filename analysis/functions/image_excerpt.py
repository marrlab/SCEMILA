import matplotlib.pyplot as plt
import math
import os
import numpy as np

CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
CLASS_COLORS = {'AML-PML-RARA': 'orange',
                'AML-NPM1': 'red',
                'AML-CBFB-MYH11': 'sienna',
                'AML-RUNX1-RUNX1T1': 'dodgerblue',
                'SCD': 'limegreen'}


def plot(
        sc_df,
        cols=16,
        dpi=150,
        show_coordinates=True,
        show_scalebar=False,
        path_save=None,
        show_patient_class=False):

    # load image array
    im_path = os.path.join(
        os.path.dirname(
            sc_df.iloc[0].im_path),
        'stacked_images.npy')
    im_ar = np.load(im_path)

    scale_size = 15 / cols
    rows = math.ceil(im_ar.shape[0] / cols)

    fig = plt.figure(
        constrained_layout=False,
        figsize=(
            15,
            (rows) *
            scale_size),
        dpi=dpi)
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)

    counter = 0
    for gs_sub in gs:

        ax = plt.subplot(gs_sub)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_coordinates:
            coordinates = CHARS[counter % cols] + \
                str(math.ceil((counter + 1) / cols))
            ax.text(5, 28, coordinates, alpha=0.4)

        if show_scalebar and counter == 0:
            # 10 mikrometer = 57,8px
            ax.plot((8, 8), (72 - 29, 72 + 29), c='k')
            ax.plot((5, 11), (72 - 29, 72 - 29), c='k')
            ax.plot((5, 11), (72 + 29, 72 + 29), c='k')

        image = im_ar[sc_df.iloc[counter]['im_id'], ...]
        ax.imshow(image)

        if show_patient_class:
            plot_color = sc_df.iloc[counter]['gt_label']
            plot_color = CLASS_COLORS[plot_color]
            circ = plt.Circle((132, 12), 4, color=plot_color)
            ax.add_patch(circ)

        counter += 1

        if(counter >= im_ar.shape[0]):
            break

        if(counter >= len(sc_df)):
            break

    plt.show()

    if path_save is not None:
        fig.savefig(path_save, bbox_inches='tight')


def plot_multipatient(
        sc_df,
        cols=16,
        dpi=150,
        show_coordinates=True,
        show_scalebar=False,
        path_save=None,
        show_patient_class=False):

    im_ar_dict = {}
    for idx, r in sc_df.iterrows():
        if r.name not in im_ar_dict.keys():
            im_ar_dict[r.name] = np.load(os.path.join(os.path.dirname(
                r['im_path']), 'processed', 'stacked_images.npy'))

    scale_size = 15 / cols
    rows = math.ceil(len(sc_df) / cols)

    fig = plt.figure(
        constrained_layout=False,
        figsize=(
            15,
            (rows) *
            scale_size),
        dpi=dpi)
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.1)

    counter = 0
    for gs_sub in gs:

        ax = plt.subplot(gs_sub)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_coordinates:
            coordinates = CHARS[counter % cols] + \
                str(math.ceil((counter + 1) / cols))
            ax.text(5, 28, coordinates, alpha=0.4)

        if show_scalebar and counter == 0:
            # 10 mikrometer = 57,8px
            ax.plot((8, 8), (72 - 29, 72 + 29), c='k')
            ax.plot((5, 11), (72 - 29, 72 - 29), c='k')
            ax.plot((5, 11), (72 + 29, 72 + 29), c='k')

        image_array_tmp = im_ar_dict[sc_df.iloc[counter].name]
        image = image_array_tmp[sc_df.iloc[counter]['im_id'], ...]

        ax.imshow(image)

        if show_patient_class:
            plot_color = sc_df.iloc[counter]['gt_label']
            plot_color = CLASS_COLORS[plot_color]
            circ = plt.Circle((132, 12), 4, color=plot_color)
            ax.add_patch(circ)

        counter += 1

        if(counter >= len(sc_df)):
            break

    plt.show()

    if path_save is not None:
        fig.savefig(path_save, bbox_inches='tight')
