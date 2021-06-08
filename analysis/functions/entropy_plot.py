import matplotlib.pyplot as plt
import beeswarm

def entropy_scatter(dataframe):
    # plot gt_label and pred_lbl separately

    classified_data_correct = dataframe.loc[dataframe['gt_label'] == dataframe['pred_lbl']]
    classified_data_false = dataframe.loc[dataframe['gt_label'] != dataframe['pred_lbl']]

    # column names will be x and y and should be swapped
    classified_data_correct_embedded = beeswarm.beeswarm_coordinates(
        classified_data_correct, val_col='entropy')
    
    classified_data_false_embedded = beeswarm.beeswarm_coordinates(
        classified_data_false, val_col='entropy')

    fig, ax = plt.subplots()

    ax.scatter(classified_data_correct_embedded.x, classified_data_correct_embedded.y)
    ax.scatter(classified_data_false_embedded.x, classified_data_false_embedded.y)

    
    # ax.set_ylim
    # ax.set_ylabel
    
    # ax.set_xticks()
    # ax.set_xticklabels()
    # ax.set_xlabel
    # ax.set_xlim
    # ax.legend()

    plt.show()
    

    