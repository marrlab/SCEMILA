from model import AMiL
import label_converter

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import os
import math
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_change_on_occlusion(
        df,
        result_folder_path,
        result_folders,
        prefix,
        lbl_conv_obj):

    # sort dataframe first by patients, then by folds (to prevent loading the
    # wrong model/the wrong data)
    df_sorted = df.sort_index(
        ascending=True).sort_values(
        by='fold',
        ascending=True).reset_index()
    label_columns = ['occlusion_' + lbl_conv_obj[x]
                     for x in range(len(lbl_conv_obj.df))]
    label_columns_solitary_pred = [
        'solitary_' + lbl_conv_obj[x] for x in range(len(lbl_conv_obj.df))]
    label_columns_solitary_pred_softmax = [
        'solitary_softmax_' + lbl_conv_obj[x] for x in range(len(lbl_conv_obj.df))]

    value_buffer = []
    pat_last, fold_last = None, None
    for idx, cell in tqdm(
            df_sorted.iterrows(), total=df.shape[0], desc="Calculate occlusion values: "):
        pat_cur, fold_cur = cell.ID, cell.fold

        # if fold has changed, load new model for proper prediction as test set
        # performance
        if not fold_cur == fold_last:
            fold_last = fold_cur

            model_folder = result_folders[fold_cur]
            # model_path = os.path.join(result_folder_path, 'state_dictmodel.pt')
            # LINE ABOVE: ERROR CAUSING INFERENCE ON TRAINING SET
            model_path = os.path.join(
                result_folder_path[:-1] + str(fold_cur), 'state_dictmodel.pt')
            model = torch.load(model_path, map_location=torch.device(device))
            print("Loaded model from: ", model_path)

        # if patient has changed, load new sc images and measure baseline
        if not pat_cur == pat_last:
            pat_last = pat_cur
            patient_feature_path = os.path.join(
                os.path.dirname(
                    cell.im_path),
                prefix +
                'bn_features_layer_7.npy')
            patient_feature_array = np.load(patient_feature_path)

            baseline, _, baseline_softmax, _ = model(
                torch.Tensor(patient_feature_array).to(device))
            baseline = baseline.cpu().detach().numpy()

        # calculate occlusion
        image_idx = list(range(patient_feature_array.shape[0]))
        image_idx.remove(cell.im_id)
        patient_feature_array_adapted = patient_feature_array[image_idx, :]

        prediction, _, prediction_softmax, _ = model(
            torch.Tensor(patient_feature_array_adapted).to(device))
        occlusion_values = prediction.cpu().detach().numpy() - baseline

        df_sorted.loc[idx, label_columns] = occlusion_values[0]

        # calculate solitary prediction. Stack image twice, to avoid squeeze()
        # reducing our tensor dimension against our will
        image_solitary = patient_feature_array[cell.im_id, :]
        image_solitary = np.stack([image_solitary, image_solitary], axis=0)
        prediction, _, prediction_softmax, _ = model(
            torch.Tensor(image_solitary).to(device))
        solitary_values = prediction.cpu().detach().numpy()
        solitary_values_softmax = F.softmax(
            prediction, dim=1).cpu().detach().numpy()
        df_sorted.loc[idx, label_columns_solitary_pred] = solitary_values[0]
        df_sorted.loc[idx,
                      label_columns_solitary_pred_softmax] = solitary_values_softmax[0]

    return df_sorted.set_index('ID')
