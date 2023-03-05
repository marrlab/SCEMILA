import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import os
import math
import random as r
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INFERENCE_SAMPLING = {
    1: 100,
    2: 100,
    4: 100,
    8: 100,
    16: 100,
    32: 100,
    64: 100,
    128: 100,
    200: 100,
    256: 100,
    500: 100
}


def calculate_subset_data(df, result_folder_path, prefix, lbl_conv_obj):

    r.seed(42)

    # sort dataframe first by patients, then by folds (to prevent loading the
    # wrong model/the wrong data)
    df_sorted = df.sort_index(
        ascending=True).sort_values(
        by='fold',
        ascending=True).reset_index()

    result_dictionary = {}
    pat_last, fold_last = None, None
    for idx, patient in tqdm(
            df_sorted.iterrows(), total=df.shape[0], desc="Calculate occlusion values: "):
        pat_cur, fold_cur = patient.ID, patient.fold
        result_dictionary[pat_cur] = {}
        result_dictionary[pat_cur]['fold'] = fold_cur
        result_dictionary[pat_cur]['filter_annotation'] = patient.filter_annotation
        result_dictionary[pat_cur]['gt'] = patient.gt_label
        result_dictionary[pat_cur]['pred'] = patient.pred_lbl
        result_dictionary[pat_cur]['inference_subsamples'] = {}

        # if fold has changed, load new model for proper prediction as test set
        # performance
        if not fold_cur == fold_last:
            fold_last = fold_cur

            model_path = os.path.join(
                result_folder_path[:-1] + str(fold_cur), 'state_dictmodel.pt')
            model = torch.load(model_path, map_location=torch.device(device))
            print("Loaded model from: ", model_path)

        # if patient has changed, load new sc images and measure baseline
        if not pat_cur == pat_last:
            pat_last = pat_cur
            patient_feature_path = os.path.join(
                patient.pat_path, 'processed', prefix + 'bn_features_layer_7.npy')
            patient_feature_array = np.load(patient_feature_path)
            im_count = patient_feature_array.shape[0]

        # calculate subset performance
        for sample_size, iterations in INFERENCE_SAMPLING.items():
            raw_buffer = []
            softmax_buffer = []
            for i in range(iterations):
                # sample patient images
                idx_available = range(im_count)
                idx_sample = r.sample(
                    idx_available, min(
                        sample_size, im_count))

                # avoid 3d input arrays
                if(len(idx_sample) == 1):
                    idx_sample = idx_sample * 2

                patient_feature_array_adapted = patient_feature_array[idx_sample, ...]
                prediction, _, prediction_softmax, _ = model(
                    torch.Tensor(patient_feature_array_adapted).to(device))
                raw_prediction = prediction.cpu().detach().numpy()
                softmax_prediction = F.softmax(
                    prediction, dim=1).cpu().detach().numpy()
                raw_buffer.append(raw_prediction)
                softmax_buffer.append(softmax_prediction)

                if(len(idx_sample) == im_count):
                    break

            # STORE IT PROPERLY AND DEBUG
            raw_buffer = np.vstack(raw_buffer)
            softmax_buffer = np.vstack(softmax_buffer)
            result_dictionary[pat_cur]['inference_subsamples'][sample_size] = (
                raw_buffer, softmax_buffer)

    return result_dictionary
