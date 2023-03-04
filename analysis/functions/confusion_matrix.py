import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import pandas as pd
import font_matching

'''Author: Matthias Hehr'''


def show_pred_mtrx(pred_mtrx, class_conversion=None, fig_size=None, normalize_rows=True, plot_values=True, 
                    show_zeros = True, show_size_values=True, reorder = None, cmap=None, dpi=100,
                    string_header = 'Confusion matrix', string_overview='# of patients', size_scale='lin',
                    capitalize = True, fontsize = 12, path_save = None, sc_df=None):

    ''' Same as above, but takes different inputs and is optimized for receiving the class_conversion dataframe:
    
    pred_mtrx:      numpy array with size(groundtruth class*predicted class). 
    class_conversion: pandas dataframe containing info about true labels and class size.
                    Needs to have columns: 'art_lbl' (network predictions), 'true_lbl' 
                    (string to display) and 'size_tot'(class size)
    fig_size:       default (6,6) or (8,6) if class_size shall be drawn, 
                    possibility to overwrite figsize for user
    normalize_rows: default True, normalizes row colors using total true class size
    plot_values:    default True, turned off vanishes raw numbers
    show_zeros:     default True, self-explaining
    show_size_values: default True, shows bar plot on the right side containing info
                    about class size
    reorder:        default None, if a list of true labels is passed, will manipulate
                    confusion matrix to have everything ordered as passed to this function.
    cmap:           pass custom color map, if none, use default.
    dpi:            default 100, self explaining
    size_scale:     defines how the bar plot on the right is scaled. Available: log, lin and None. If None, then a linear scale,
                    starting from zero will be applied.
    string_overview:string displayed over right overview plot
    string_header:  change confusion matrix caption
    capitalize:     define if the legend starts should be capitalized or not
    fontsize:       self explaining
    path_save:      path to store the confusion matrix
    '''
    
    if class_conversion is None:
        raise NameError("no class conversion matrix given. Not gonna work here.")
    class_conversion = class_conversion.copy().sort_values(by=['art_lbl'], ascending=True)
    class_conversion = class_conversion.set_index('art_lbl')

    # deal with manipulated order of confusion matrix
    manipulated_positions = []
    for i, row in class_conversion.iterrows():
        if not reorder is None:
            true_lbl = row['true_lbl']
            manipulated_positions.append(reorder.index(true_lbl))
        else:
            manipulated_positions.append(len(manipulated_positions))

    class_conversion['manipulated_order'] = manipulated_positions
    pred_mtrx = manipulate_matrix(pred_mtrx, class_conversion)

    class_conversion = class_conversion.sort_values(by=['manipulated_order'], ascending=True)
    class_conversion = class_conversion.set_index('manipulated_order')
    


    # set up parameters
    class_size=len(class_conversion)
    if(fig_size is None):
        fig_size=(12,7)
    gridspec_width=3
    width_rat=[5,2,3]

    # convert dictionarys into arrays to prevent wrong order of labels/bars
    capitalize_func = lambda x: [y[0].upper() + y[1:] for y in x]
    name_list = []
    freq_list = []
    for el in range(class_size):
        name_list.append(class_conversion.at[el, 'true_lbl'])
        freq_list.append(class_conversion.at[el, 'size_tot'])

    if (capitalize):
        name_list = capitalize_func(name_list)

    
    # initiate subplot layout, depending on wether frequencys shall be shown
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, constrained_layout=True, figsize=fig_size, dpi=dpi, gridspec_kw={'width_ratios':width_rat, 'wspace':0.1}, sharey=False)

    #normalize pred_mtrx for rows
    pred_mtrx_copy = pred_mtrx.astype(float)
    if normalize_rows:
        for row in range(class_size):
            if (not np.sum(pred_mtrx_copy[row, :]) == 0):    #avoid division by zero
                pred_mtrx_copy[row, :] = pred_mtrx_copy[row, :]/np.sum(pred_mtrx_copy[row, :])

    if(cmap is None):
        cmap='Greys'
    im = ax0.imshow(pred_mtrx_copy, cmap=cmap)

    name_list = ['Control' if x=='SCD' else x for x in name_list]
    name_list = [font_matching.edit(x) for x in name_list]

    # set up proper axis labelling/ticks
    ax0.set_xticks(np.arange(class_size))
    ax0.set_yticks(np.arange(class_size))
    ax0.set_xticklabels(name_list)
    ax0.set_yticklabels(name_list)
    ax0.set_xlabel("Network prediction", fontdict=None, labelpad=None, fontsize=fontsize)
    ax0.set_ylabel("Groundtruth", fontdict=None, labelpad=None, fontsize=fontsize)
    ax0.set_title(string_header, fontsize=fontsize)
    ax0.set_ylim(class_size-0.5, -0.5)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax0.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    # Iterate over data dimensions and create text annotations.
    if(plot_values):
        for i in range(class_size):
            for j in range(class_size):
                if(not show_zeros) and pred_mtrx[i,j]==0:
                    continue
                col_font='k'
                if(pred_mtrx_copy[i,j] > 0.4 and cmap == 'Greys'):
                    col_font='w'
                text = ax0.text(j, i, pred_mtrx[i, j], fontsize=fontsize,
                    ha="center", va="center", color=col_font)

    # Plot data frequencys if values are given
    
    # Calculate border until when labels shall be drawn outside of bar
    if(show_size_values):

        if(size_scale=='log'):
            log_min = math.log10(min(freq_list)*0.8)
            log_max = math.log10(max(freq_list)*1.05)
            log_border = 10**(((log_max-log_min)*0.4)+log_min)            #for logarithmic plotting
        
        if(size_scale=='lin'):
            log_min = min(freq_list)*0.8
            log_max = max(freq_list)*1.1
            log_border = ((log_max-log_min)*0.4)+log_min

        if(size_scale is None):
            log_min = 0
            log_max = max(freq_list)*1.1
            log_border = log_max*0.4

        for el in range(class_size):
            size = freq_list[el]

            if(size > log_border):
                ax1.text(size, el, str(size) + " ", va='center', ha='right', color='w', fontsize=fontsize)
            else:
                ax1.text(size, el, " " + str(size), va='center', ha='left', color='k', fontsize=fontsize)
    
    # patient count plot
    ax1.barh(range(class_size), freq_list, color='black')   
    ax1.set_yticks(np.arange(len(class_conversion)))
    

    ax1.set_xlim(0,max(freq_list)*1.05)
    ax1.set_yticks([])
    ax1.set_ylim(class_size-0.5, -0.5)
    
    ax1.set_title("Patients", fontdict=None, fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    # image count plot
    image_counts = []
    for el in list(class_conversion.true_lbl):
        images_of_class = sc_df.loc[sc_df['gt_label'] == el]
        image_counts.append(len(images_of_class))
    ax2.barh(range(class_size), image_counts, color='black')   

    ax2.set_xlim(0,max(image_counts)*1.05)
    ax2.set_yticks([])
    ax2.set_ylim(class_size-0.5, -0.5)
    
    ax2.set_title("Single cell images", fontdict=None, fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    if(show_size_values):

        log_min = 0
        log_max = max(image_counts)
        log_border = log_max*0.66

        for el in range(class_size):
            size = image_counts[el]

            if(size > log_border):
                ax2.text(size, el, str(size) + " ", va='center', ha='right', color='w', fontsize=fontsize)
            else:
                ax2.text(size, el, " " + str(size), va='center', ha='left', color='k', fontsize=fontsize)


    plt.show()

    if not path_save is None:
        fig.savefig(path_save, bbox_inches='tight')

def manipulate_matrix(pred_mtrx, conversion):
    pred_mtrx_reordered = pred_mtrx.copy()

    for pos_x in range(pred_mtrx.shape[1]):
        for pos_y in range(pred_mtrx.shape[0]):
            reordered_x = conversion.at[pos_x, 'manipulated_order']
            reordered_y = conversion.at[pos_y, 'manipulated_order']
            pred_mtrx_reordered[reordered_y, reordered_x] = pred_mtrx[pos_y, pos_x]

    return pred_mtrx_reordered
    
def save_plot(PATH):
    plt.savefig(fname=PATH)






##### Below is analysis to extract numbers from confusion matrix
def get_classwise_values(conf_matrix):
    
    # get precision, recall, selectivity for each class
    class_count = conf_matrix.shape[0]
    classwise_precision = np.zeros(class_count)
    classwise_recall = np.zeros(class_count)
    classwise_selectivity = np.zeros(class_count)
    classwise_f1 = np.zeros(class_count)
    classwise_accuracy = np.zeros(class_count)
    
    for el in range(class_count):
        tp, fp, tn, fn = 0,0,0,0
        
        tp = conf_matrix[el, el]
        fp = np.sum(conf_matrix[:, el])-tp
        fn = np.sum(conf_matrix[el, :])-tp
        tn = np.sum(conf_matrix)-fp-fn-tp
        
        classwise_precision[el] = get_precision(tp, fp)
        classwise_recall[el] = get_recall(tp, fn)
        classwise_selectivity[el] = get_selectivity(tn, fp)
        classwise_f1[el] = get_f1(tp, fp, fn)
        classwise_accuracy[el] = get_acc_classwise(tp, fp)
        
    return classwise_precision, classwise_recall, classwise_selectivity, classwise_f1, classwise_accuracy


def get_classwise_values_as_df(conf_matrix, lbl_conv):
    df = pd.DataFrame([], columns=['class', 'prec', 'rec', 'f1', 'sens', 'spec']).set_index('class')
    true_names = lbl_conv.df.true_lbl
    for el in range(len(true_names)):
        tp = conf_matrix[el, el]
        fp = np.sum(conf_matrix[:, el])-tp
        fn = np.sum(conf_matrix[el, :])-tp
        tn = np.sum(conf_matrix)-fp-fn-tp
        label = true_names[el]

        data = (get_precision(tp, fp), get_recall(tp, fn), get_f1(tp, fp, fn), get_recall(tp, fn), get_selectivity(tn, fp))
        df.loc[label] = data
    
    return df

def get_fold_statistics(conf_matrices_dict, lbl_conv, metric='f1'):
    c_data = None
    for fold, confusion_data in conf_matrices_dict.items():
        df = get_classwise_values_as_df(confusion_data, lbl_conv)
        new_metric_string = '{}_{}'.format(metric, fold)

        if c_data is None:
            c_data = df[metric].to_frame().rename(columns={metric:new_metric_string})
        else:
            c_data[new_metric_string] = df[metric]

    mean = c_data.mean(axis=1)
    std = c_data.std(axis=1)

    c_data['mean'] = mean
    c_data['std'] = std
    
    return c_data

    
        
def get_accuracy(conf_matrix):
    tp = sum([conf_matrix[i,i] for i in range(conf_matrix.shape[0])])
    return tp/np.sum(conf_matrix)

def get_precision(true_pos, false_pos):
    if(true_pos + false_pos == 0):
        return 0
    val = true_pos/(true_pos+false_pos)
    return val

def get_recall(true_pos, false_neg):
    if(true_pos + false_neg == 0):
        return 0
    val = true_pos/(true_pos+false_neg)
    return val

def get_selectivity(true_neg, false_pos):
    ''' equal to specificity '''
    if(true_neg + false_pos == 0):
        return 0    
    val = true_neg/(true_neg+false_pos)
    return val

def get_f1(true_pos, false_pos, false_neg):
    precision = get_precision(true_pos, false_pos)
    recall = get_recall(true_pos, false_neg)

    if(precision+recall == 0):
        return 0
    return 2*(precision*recall)/(precision+recall)

def get_acc_classwise(true_pos, false_pos):
    if(true_pos+false_pos == 0):
        return 0
    return true_pos/(true_pos+false_pos)
