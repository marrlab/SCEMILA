import matplotlib.pyplot as plt
import os
import font_matching


def plot(df_row, reorder, path_save=None):
    fig, ax = plt.subplots(figsize=(5, 3))

    y = 4
    ypos = []
    for el in reorder:
        x_val = df_row['mil_prediction_{}'.format(el)]
        ax.barh(y=y, width=x_val, color='w', edgecolor='k', linewidth=1)

        ypos.append(y)
        y -= 1

    ax.set_yticks(ypos)

    reorder_labels = [font_matching.edit(x) for x in reorder]
    ax.set_yticklabels(reorder_labels, fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.set_xlabel('Algorithm output')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(
        'Prediction for patient {}'.format(
            os.path.basename(
                df_row.pat_path)),
        fontsize=12)

    # plot groundtruth label
    gt = df_row.gt_label
    gt_y = ypos[reorder.index(gt)]
    gt_x = df_row['mil_prediction_{}'.format(gt)] + 0.05
    ax.scatter(
        gt_x,
        gt_y,
        marker='o',
        s=125,
        color='w',
        edgecolor='k',
        linewidth=1,
        label='Groundtruth')

    # plot predicted label
    pred = df_row.pred_lbl
    pred_y = ypos[reorder.index(pred)]
    pred_x = df_row['mil_prediction_{}'.format(pred)] + 0.05
    ax.scatter(
        pred_x,
        pred_y,
        marker='x',
        s=125,
        color='k',
        linewidth=1,
        label='Prediction')

    leg = ax.legend(
        title='Patient label',
        title_fontsize=12,
        fontsize=12,
        edgecolor='gainsboro',
        shadow=True)
    leg._legend_box.align = "left"
    plt.show()

    if path_save is not None:
        fig.savefig(path_save, bbox_inches='tight')
