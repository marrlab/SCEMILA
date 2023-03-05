from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


def plot(patient_dataframe, true_list, save_in=None):

    roc_df = []
    for idx, row in patient_dataframe.iterrows():
        has_true_label = row.gt_label in true_list
        perc = row[['mil_prediction_' + x for x in true_list]].sum()
        roc_df.append((has_true_label, perc))

    df_packed = pd.DataFrame(roc_df, columns=['is_true', 'percentage'])
    df_packed.sort_values(by=['percentage'], ascending=False, inplace=True)

    fpr, tpr, _ = metrics.roc_curve(
        df_packed['is_true'], df_packed['percentage'])
    roc_auc = metrics.auc(fpr, tpr)

    spec = 1 - fpr
    fig = plt.figure(figsize=(7, 3), dpi=150)
    g = fig.add_gridspec(1, 2, wspace=0.4)

    ax = fig.add_subplot(g[0])
    ax.plot(spec, tpr, label="resnext", color='k')
    ax.plot((1, 0), (0, 1), color='lightgray')
    ax.set_xlabel('Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title('{}\ndetection'.format(true_list), fontsize=12)
    ax.text(0.1, 0.1, "AUC: {:.3f}".format(roc_auc), fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax = fig.add_subplot(g[1])
    prec, rec, _ = metrics.precision_recall_curve(
        df_packed['is_true'], df_packed['percentage'])
    ax.plot(rec, prec, label="resnext", color='k')
    ax.plot((1, 0), (0, 1), color='lightgray')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title('{}\nsdetection'.format(true_list))
    ax.text(
        0.1,
        0.1,
        "AUC: {:.3f}".format(
            metrics.auc(
                rec,
                prec)),
        fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()
    if save_in is not None:
        fig.savefig(save_in, bbox_inches='tight')


def return_roc_data(patient_dataframe, true_list):

    roc_df = []
    for idx, row in patient_dataframe.iterrows():
        has_true_label = row.gt_label in true_list
        perc = row[['mil_prediction_' + x for x in true_list]].sum()
        roc_df.append((has_true_label, perc))

    df_packed = pd.DataFrame(roc_df, columns=['is_true', 'percentage'])
    df_packed.sort_values(by=['percentage'], ascending=False, inplace=True)

    fpr, tpr, _ = metrics.roc_curve(
        df_packed['is_true'], df_packed['percentage'])
    spec = 1 - fpr
    spec_sens_auc = metrics.auc(fpr, tpr)

    prec, rec, _ = metrics.precision_recall_curve(
        df_packed['is_true'], df_packed['percentage'])
    prec_rec_auc = metrics.auc(rec, prec)

    return spec, tpr, spec_sens_auc, prec, rec, prec_rec_auc
