import base64
from io import BytesIO
import numpy as np
from tqdm import tqdm
import os
from PIL import Image


def embeddable_image(array, idx):
    img_data = array[idx, ...]
    image = Image.fromarray(
        img_data, mode='RGB').resize(
        (64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def map_images_to_dataframe(df):
    df = df.reset_index()
    df['image'] = [None] * len(df)
    df = df.sort_values(by=['ID', 'im_tiffname'])

    current_pat_id = ''

    for row_idx in tqdm(range(len(df)), desc="Preload single cell images"):

        pat_id = df.loc[row_idx].ID
        if pat_id != current_pat_id:
            # load numpy array
            patient_basepath = os.path.dirname(df.loc[row_idx].im_path)

            patient_images_path = os.path.join(
                patient_basepath, 'stacked_images.npy')
            patient_images = np.load(patient_images_path)
            current_pat_id = pat_id

        im_pos = int(df.loc[row_idx].im_tiffname.split('.')[0])
        df.loc[row_idx, 'image'] = embeddable_image(patient_images, im_pos)

    return df
