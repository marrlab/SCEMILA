import seaborn as sns
import matplotlib.pyplot as plt


def entropy_plot(dataframe):
    df_tmp = dataframe.copy()
    df_tmp['classification'] = df_tmp['gt_label'] == df_tmp['pred_lbl']
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = sns.boxplot(ax=ax, data=df_tmp, x="classification", y="entropy", hue="quality_category")
    # ax = sns.swarmplot(ax=ax, data=df_tmp, x="classification", y="entropy", hue="quality_category")
    ax = sns.swarmplot(ax=ax, data=df_tmp, x="classification", y="entropy")
    ax.set(ylabel="Entropy")
    ax.set(xlabel="Classification")


def entropy_vs_myb(dataframe):
    df_tmp = dataframe.copy()
    df_tmp['classification'] = df_tmp['gt_label'] == df_tmp['pred_lbl']
    fig, ax = plt.subplots(figsize=(8, 6))

    df_true = df_tmp.loc[df_tmp['classification']]
    ax.scatter(df_true.myb_annotated, df_true.entropy, label='True prediction')
    df_false = df_tmp.loc[~df_tmp['classification']]
    ax.scatter(
        df_false.myb_annotated,
        df_false.entropy,
        label='False prediction')
    ax.set(ylabel="Entropy")
    ax.set(xlabel="Myeloblast percentage")
    ax.legend(title='Percent of myeloblasts in differential blood count',
              bbox_to_anchor=(1, 0., 0.5, 1), loc=10)
