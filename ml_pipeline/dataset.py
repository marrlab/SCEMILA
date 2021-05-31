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



##### Path to pickle objects for quicker data loading
dataset_defined = False                                                                 # used to make sure define_dataset is called before creating new dataset object

def process_label(str_in, pat_id):
    '''This function allows to change string labels, e.g. sum up different classes in one
    major class. If str_in is returned, the default class will be used.
    '''

    # three class
    # if str_in in ['AML-INV16', 'AML-NPM1', 'AML-PML-RARA', 'AML-t(8;21)', 'AML-other']:
    #     return 'AML'
    # else:
    #     return str_in

    # regular full-class classification
    return str_in

def set_dataset_path(path):
    ''' Pass on path to locate the data '''
    global path_data
    path_data = path


##### Load and filter dataset according to custom criteria, before Dataset initialization
def define_dataset(num_folds = 5, prefix_in=None, label_converter_in=None,
                    filter_nonvisible=-1):
        '''function needs to be called before constructing datasets. Gets an overview over the entire dataset, does filtering
        according to parameters / exclusion criteria, and splits the data automatically.
        - num_folds: split dataset into n folds
        - prefix_in: choose different set of extracted features, by adapting here
        - label_converter_in: needs label_converter created in launcher.py to function and convert string to integer labels
        - filter_nonvisible:filter by myeloblast fraction in patients. Filters patients with less or equal (<=) myeloblast fraction!
                            A few special values: 
                                if set to (-2), no filtering happens at all
                                if set to (-1), no filtering happens due to myeloblast count, but due to exclusion for other reasons
                                from (0) on, filtering kicks in.
        - exclude_below: exclude class, if it has less than n patients'''
        
        global dataset_defined, prefix, mil_distribution, mil_mutations, label_conv_obj

        prefix = prefix_in
        label_conv_obj = label_converter_in

        # load patient data
        df_data_master = pd.read_csv('{}/mll_data_master_pseudo.csv'.format(path_data)).set_index('pseudonym')

        # iterate over all patients in the df_data_master sheet
        merge_dict_processed = {}
        for idx, r in df_data_master.iterrows():

            # define filter criterion by which to filter the patients by annotation
            annotations_exclude_by = ['pb_myeloblast']
                
            # iterate over every patient
            for pat in val:
                pat_id = os.path.basename(pat)

                # allow for filtering deactivation
                if not (filter_nonvisible == -2):

                    # filter if patient not found
                    id_loc = mll_diff_annotation[mll_diff_annotation['ID'] == id_truncated]
                    if len(id_loc) == 0:
                        print("Pat not found: ", pat_id)
                        continue

                    # filter if annotation data for the patient is not found
                    is_annotated = id_loc['Pb Total'] > 0
                    annotate_exclude_by = sum(sum(id_loc[annotations_exclude_by].values))
                    if not is_annotated.all():
                        print("No annotation: ", pat_id)
                        continue
                        
                    # filter if patient has not enough cells (only if an AML patient)
                    if annotate_exclude_by <= filter_nonvisible and (not key in ['stem cell donor', 'non-AML']):
                        print("Not enough malign cells, exclude: ", annotate_exclude_by, " ", pat_id)
                        continue

                    # filter if manual assessment revealed flaws
                    quality_exclude = mll_data_assessment_exclude[mll_data_assessment_exclude['PatID'] == id_truncated]
                    ##### first: not found in file - filter in the future, if all patients have been assessed
                    if not len(quality_exclude) == 0:
                        # print("AGAIN_Pat. not found: ", pat_id)
                        # continue
                        
                        ##### second: if in hard_exclude_criteria: 
                        if not pd.isna(quality_exclude.cat_exclude_quality.iloc[0]):
                            print("Bad slide quality, exclude: ", pat_id)
                            continue

                        ##### third: if in optional_exclude_criteria: 
                        # if not pd.isna(quality_exclude.cat_optional_takeout.iloc[0]):
                        #     print("Bad slide quality, exclude: ", pat_id)
                        #     continue


                # enter patient into label converter
                label = process_label(key, pat_id)
                if label is None:
                    continue
                
                # store patient for later loading
                if not label in merge_dict_processed.keys():
                    merge_dict_processed[label] = []
                merge_dict_processed[label].append(pat)
            
        # split dataset
        dataset_defined = True
        data_split.split_in_folds(merge_dict_processed, num_folds)


##### Actual dataset class
class dataset(Dataset):

    '''MLL mil dataset class. Can be used by pytorch DataLoader '''

    def __init__(self, folds=range(3), aug_im_order=True, split=None):
        '''dataset constructor. Accepts parameters:
        - folds: list of integers or integer in range(NUM_FOLDS) which are set in beginning of this file.
                Used to define split of data this dataset should countain, e.g. 0-7 for train, 8 for val, 
                9 for test
        - aug_im_order: if True, images in a bag are shuffled each time during loading
        - split: store information about the split within object'''
        
        if(not dataset_defined):
            raise NameError('No dataset defined. Use define_dataset before initializing dataset class')

        self.aug_im_order = aug_im_order

        ##### grab data split for corresponding folds
        self.data = data_split.return_folds(folds)
        self.paths = []
        self.labels = []

        # reduce the hard drive burden by storing features in a dictionary in RAM, once they have been loaded later
        self.features_loaded = {}
        
        # enter paths and corresponding labels in self.data
        for key, val in self.data.items():
            self.paths.extend(val)
            
            label_conv_obj.add(key, len(val), split=split)
            label = label_conv_obj[key]
            self.labels.extend([label]*len(val))

            


    def __len__(self):
        '''returns amount of images contained in object'''
        return len(self.paths)


    def __getitem__(self, idx):
        '''returns specific item from this dataset'''

        # grab images, patient id and label
        path = self.paths[idx]
        
        # only load if object has not yet been loaded
        if (not path in self.features_loaded):
            bag = np.load(os.path.join(path, 'processed', prefix + 'bn_features_layer_7.npy'))
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
