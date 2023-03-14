from tqdm import tqdm
import numpy as np
import pickle as pkl
import pandas as pd
import os
import sys
from scipy.stats import entropy

import label_converter


def generate_tiffname(x): return '0' * (3 - len(str(x))) + str(x) + '.TIF'


has_multiatt = False


def load_dataframes(
        folder_list,
        basepath,
        prefix,
        folder_dataset,
        load_sc_features=True):
    '''This function loads...
    All testing data object files.
    All feature vectors for the UMAP embeddings.

    Returns:
    - dataframe with all single cell images including their features
    - dataframe with all patient data
    '''

    global has_multiatt

    # load label converter for further processing
    lbl_conv_obj = label_converter.LabelConverter(
        path_preload=os.path.join(
            basepath, folder_list[0], 'class_conversion.csv'))
    patient_master_dataframe = pd.read_csv(
        '{}/metadata.csv'.format(folder_dataset)).set_index('patient_id')

    datapoints = []
    temporary_data_cache = {}

    ''' Load all Patients into dataframe.
    Dataframe has columns:
    - pat_ID (identifier)
    - pat_path (full path to patient)
    - fold
    - groundtruth label
    - MIL loss
    - prediction_vector '''
    for f in folder_list:

        if(len(folder_list) > 1):
            f_fold = int(f[-1:])
        else:
            f_fold = 0

        ''' Below: Load the data from the data matrix. '''
        f_full_target = os.path.join(basepath, f, 'testing_data.pkl')
        if not os.path.exists(f_full_target):
            continue

        f_stream = open(f_full_target, 'rb')
        pkl_obj = pkl.load(f_stream)
        f_stream.close()

        for entity, patients in pkl_obj.items():
            for pat_path, pat_data_packed in patients.items():
                pat_id = os.path.basename(pat_path)
                pat_attention_raw, pat_attention_softmax, _, pat_prediction_vector, pat_loss, _ = pat_data_packed
                pat_prediction_argmax = np.argmax(pat_prediction_vector)
                pat_entropy = entropy(
                    pat_prediction_vector,
                    base=len(pat_prediction_vector))
                
                pat_myb_share = patient_master_dataframe.loc[pat_id,
                                                             'pb_myeloblast']
                pat_pmc_share = patient_master_dataframe.loc[pat_id,
                                                             'pb_promyelocyte']
                pat_myc_share = patient_master_dataframe.loc[pat_id,
                                                             'pb_myelocyte']
                pat_datapoint = [pat_id,
                                 pat_path,
                                 f_fold,
                                 lbl_conv_obj[entity],
                                 lbl_conv_obj[int(pat_prediction_argmax)],
                                 pat_loss,
                                 pat_entropy,
                                 pat_myb_share,
                                 pat_myb_share + pat_pmc_share + pat_myc_share]
                pat_datapoint.extend(pat_prediction_vector)

                # enter data in dataframe
                datapoints.append(pat_datapoint)
                temporary_data_cache[pat_id] = (
                    pat_attention_raw, pat_attention_softmax, pat_prediction_vector)

    # determine if the algorithm was set to multi attention, or single
    # attention
    has_multiatt = pat_attention_raw.shape[0] > 1

    columns_df = [
        'ID',
        'pat_path',
        'fold',
        'gt_label',
        'pred_lbl',
        'MIL loss',
        'entropy',
        'myb_annotated',
        'filter_annotation']
    columns_df.extend(['mil_prediction_' + lbl_conv_obj[x]
                      for x in range(len(lbl_conv_obj.df))])
    patient_dataframe = pd.DataFrame(
        datapoints, columns=columns_df).set_index('ID')

    ''' Load all single cell data into dataframe.
    Dataframe contains:
    - Patient ID ('ID')
    - fold
    - mil label
    - mil prediction
    - attention with softmax (for all entities)
    - attention without softmax (for all entities)
    - 12800 single cell features '''
    patientwise_single_cell_dataframes = []
    columns_df = ['ID', 'fold', 'gt_label']
    columns_df.extend(['mil_prediction_' + lbl_conv_obj[x]
                      for x in range(len(lbl_conv_obj.df))])
    if has_multiatt:
        att_softmax_columns = ['att_softmax_' + lbl_conv_obj[x]
                               for x in range(len(lbl_conv_obj.df))]
        att_raw_columns = ['att_raw_' + lbl_conv_obj[x]
                           for x in range(len(lbl_conv_obj.df))]
    else:
        att_softmax_columns = ['att_softmax']
        att_raw_columns = ['att_raw']

    for row_idx in tqdm(range(len(patient_dataframe))):
        pat_datapoint = patient_dataframe.iloc[row_idx]

        pat_path = pat_datapoint.pat_path
        pat_id = pat_datapoint.name
        pat_path = pat_datapoint.pat_path
        pat_fold = pat_datapoint.fold
        pat_groundtruth = pat_datapoint.gt_label
        pat_prediction_vector = temporary_data_cache[pat_id][2]
        pat_attention_softmax = temporary_data_cache[pat_id][1]
        pat_attention_raw = temporary_data_cache[pat_id][0]
        num_single_cells = len(pat_attention_softmax[0])

        pat_data_common = [pat_id, pat_fold, pat_groundtruth]
        pat_data_common.extend(pat_prediction_vector)
        pat_sc_dataframe = [pat_data_common] * num_single_cells
        pat_sc_dataframe = pd.DataFrame(pat_sc_dataframe, columns=columns_df)
        pat_sc_dataframe['im_tiffname'] = [
            generate_tiffname(x) for x in range(num_single_cells)]
        pat_sc_dataframe['im_path'] = [
            os.path.join(
                pat_path,
                generate_tiffname(x)) for x in range(num_single_cells)]
        pat_sc_dataframe['im_id'] = range(num_single_cells)

        pat_sc_dataframe[att_softmax_columns] = pd.DataFrame(
            np.swapaxes(pat_attention_softmax, 0, 1))
        pat_sc_dataframe[att_raw_columns] = pd.DataFrame(
            np.swapaxes(pat_attention_raw, 0, 1))

        if load_sc_features:
            pat_features = np.load(
                os.path.join(
                    pat_path,
                    '{}bn_features_layer_7.npy'.format(prefix)))
            ft_dims = pat_features.shape
            pat_features_flattened = pat_features.reshape(
                (ft_dims[0], ft_dims[1] * ft_dims[2] * ft_dims[3]))            # keeps image dimension
            pat_sc_feature_dataframe = pd.DataFrame(
                pat_features_flattened, columns=[
                    str(x) for x in range(12800)])
            pat_sc_dataframe_full = pat_sc_dataframe.join(
                pat_sc_feature_dataframe)

            patientwise_single_cell_dataframes.append(pat_sc_dataframe_full)
        else:
            patientwise_single_cell_dataframes.append(pat_sc_dataframe)

    # merge all the dataframes
    whole_sc_dataframe = pd.concat(patientwise_single_cell_dataframes, axis=0)
    mll_annotation_dataframe = pd.read_csv(
        'suppl_data/image_annotation_master.csv').set_index(['ID', 'im_tiffname'])
    whole_sc_dataframe = whole_sc_dataframe.join(
        mll_annotation_dataframe, on=[
            'ID', 'im_tiffname'], how='left').set_index('ID')

    return lbl_conv_obj, patient_dataframe, whole_sc_dataframe


def extract_confusion_matrix(sc_df, lbl_conv_obj):

    num_classes = len(lbl_conv_obj.df)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int16)

    for idx, r in sc_df.iterrows():
        groundtruth = lbl_conv_obj[r.gt_label]
        prediction = lbl_conv_obj[r.pred_lbl]

        confusion_matrix[groundtruth, prediction] += 1

    return confusion_matrix


def get_raw_attention_column(entity):

    global has_multiatt

    out_string = 'att_raw'
    if has_multiatt:
        out_string = out_string + '_' + entity

    return out_string


def get_softmax_attention_column(entity):

    global has_multiatt

    out_string = 'att_softmax'
    if has_multiatt:
        out_string = out_string + '_' + entity

    return out_string


def get_solitary_column(entity):

    out_string = 'solitary'
    out_string = out_string + '_' + entity

    return out_string
