import numpy as np
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys


import data_split
import label_converter
import os
import pandas as pd


# Path to pickle objects for quicker data loading
# used to make sure define_dataset is called before creating new dataset object
dataset_defined = False


def process_label(row):
    '''process_label allows the user to modify the classification task.
    This can be achieved by changing the lines of code below, looking up 
    the label in the master dataframe.

    Accepts:
    - row (pd.Series):
        All the data about the patient to be sampled.

    Returns:
    - lbl_out (String):
        Label which the patient should be classified as.
    '''

    # classification for the bag label: SCD vs 4 different types of AML
    lbl_out = row.bag_label

    # alternative labelling, e.g. by sex:
    # lbl_out = ['female', 'male'][int(row.sex_1f_2m)-1]

    return lbl_out


def set_dataset_path(path):
    '''Define the path to the dataset before loading the data.
    This function is called in run_pipeline.py, to automatically
    adjust to changes in the actual script.

    Accepts:
    - path (path) to the dataset. 

    Returns:
    - Nothing
    '''

    global path_data
    path_data = path


# Load and filter dataset according to custom criteria, before Dataset initialization
def define_dataset(num_folds=5, prefix_in=None, label_converter_in=None,
                   filter_diff_count=0, filter_quality_major_assessment=True, filter_quality_minor_assessment=True):
    '''This function needs to be called before constructing datasets. Gets an overview over the entire dataset, does filtering
    according to parameters / exclusion criteria, and splits the data automatically.

    Accepts:
    - num_folds (int): 
        split dataset into n folds
    - prefix_in (String): 
        Vary the features used for the algorithm. Default is fnl34_, which are the features used for the paper
    - label_converter_in (LabelConverter): 
        This object stores the amount of patients for each label, and allows for quick and easy conversion.
        For more info, see label_converter.py.
    - filter_diff_count (integer): 
        filter if patient has less than this amount (e.g. '19' for 19%) myeloblasts. 
        Set to 0, if no filtering should be applied here.
    - filter_quality_major_assessment: exclude slides with unacceptable slide quality
    - filter_quality_minor_assessment: exclude slides with sub-standard quality, which are still ok
        if not enough data is available

    Returns:
    - Nothing. Responsible for filtering the dataset appropriately.
    '''

    global dataset_defined, prefix, mil_distribution, mil_mutations, label_conv_obj

    prefix = prefix_in
    label_conv_obj = label_converter_in

    # load patient data
    df_data_master = pd.read_csv(
        '{}/mll_data_master_pseudo.csv'.format(path_data)).set_index('pseudonym')

    print("")
    print("Filtering the dataset...")
    print("")

    # iterate over all patients in the df_data_master sheet
    merge_dict_processed = {}
    for idx, row in df_data_master.iterrows():

        # filter if patient has not enough malign cells (only if an AML patient)
        # define filter criterion by which to filter the patients by annotation
        annotations_exclude_by = ['pb_myeloblast',
                                  'pb_promyelocyte', 'pb_myelocyte']
        annotation_count = sum(row[annotations_exclude_by])
        if annotation_count < filter_diff_count and (not row['bag_label'] == 'SCD'):
            print("Not enough malign cells, exclude: ", row.name,
                  " with ", annotation_count, " malign cells ")
            continue

        # filter if manual assessment revealed major flaws. If this cell contains N/A, then we don't exclude
        keep_row = pd.isnull(row['examine_exclude'])

        # filter if the patient has known bad sample quality
        if not keep_row and filter_quality_major_assessment:
            print("Major flaws in slide quality, exclude: ", row.name, " ")
            continue

        # filter if manual assessment revealed *minor* flaws. If this cell contains N/A, then we don't exclude
        keep_row = pd.isnull(row['examine_optional_exclude'])

        # filter if the patient has known bad sample quality
        if not keep_row and filter_quality_minor_assessment:
            print("Minor flaws in slide quality, exclude: ", row.name, " ")
            continue

        # enter patient into label converter
        label = process_label(row)
        if label is None:
            continue

        # store patient for later loading
        if not label in merge_dict_processed.keys():
            merge_dict_processed[label] = []
        patient_path = os.path.join(
            path_data, 'data', row['bag_label'], row.name)
        merge_dict_processed[label].append(patient_path)

    # split dataset
    dataset_defined = True
    data_split.split_in_folds(merge_dict_processed, num_folds)
    print("Data filtering complete.")
    print("")


# Actual dataset class
class MllDataset(Dataset):

    '''MLL mil dataset class. Works with the pytorch dataloader.'''

    def __init__(self, folds=range(3), aug_im_order=True, split=None, patient_bootstrap_exclude=None):
        '''
        Accepts:
        - folds (list of Integers): 
            All the folds which should be part of this dataset.
        - aug_im_order (Boolean):
            Augment image order randomly upon loading of the patient.
            This improves the fitting of the attention network
        - split (String): 
            Required to store the proper sizes of training, validation and test 
            in the label_converter object.
        - patient_bootstrap_exclude (integer):
            Allow for bootstrapping: Exclude specific individual patients
            to check for algorithm stability. Leave None, and no bootstrapping
            takes place.

        Returns:
        - Nothing
        '''

        if(not dataset_defined):
            raise NameError(
                'No dataset defined. Use define_dataset before initializing dataset class')

        self.aug_im_order = aug_im_order

        # grab data split for corresponding folds
        self.data = data_split.return_folds(folds)
        self.paths, self.labels = [], []

        # reduce the hard drive burden by storing features in a dictionary in RAM, as they will be used again
        self.features_loaded = {}

        # enter paths and corresponding labels in self.data
        for key, val in self.data.items():
            if not patient_bootstrap_exclude is None:
                if(0 <= patient_bootstrap_exclude < len(val)):
                    path_excluded = val.pop(patient_bootstrap_exclude)
                    patient_bootstrap_exclude = -1
                    print("Bootstrapping. Excluded: ", path_excluded)
                else:
                    patient_bootstrap_exclude -= len(val)

            self.paths.extend(val)

            label_conv_obj.add(key, len(val), split=split)
            label = label_conv_obj[key]
            self.labels.extend([label]*len(val))

    def __len__(self):
        '''
        Returns:
        - Amount of patients in this dataset
        '''
        return len(self.paths)

    def __getitem__(self, idx):
        '''
        Function to sample data.
        Allows for augmentation.

        Accepts:
        - idx (Integer): Identifier for specific patient

        Returns:
        - bag (torch.Tensor): features for all images of this patient
        - label_regular (torch.Tensor): one-hot encoded patient label
        - pat_id (String): Individual patient identifier
        '''

        # grab images, patient id and label
        path = self.paths[idx]

        # only load if object has not yet been loaded
        if (not path in self.features_loaded):
            bag = np.load(os.path.join(path, 'processed',
                          prefix + 'bn_features_layer_7.npy'))
            self.features_loaded[path] = bag
        else:
            bag = self.features_loaded[path].copy()

        label = self.labels[idx]
        pat_id = path

        # shuffle features by image order in bag, if desired
        if(self.aug_im_order):
            num_rows = bag.shape[0]
            new_idx = torch.randperm(num_rows)

            bag = bag[new_idx, :]

        # prepare labels as one-hot encoded
        label_onehot = torch.zeros(len(self.data))
        label_onehot[label] = 1

        label_regular = torch.Tensor([label]).long()

        return bag, label_regular, pat_id
