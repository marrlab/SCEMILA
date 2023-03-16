import umap
import pickle as pkl
import os
from sklearn.preprocessing import StandardScaler

FEATURES = [str(x) for x in range(12800)]
PATH_EMBEDDINGS = 'suppl_data/embeddings'

scaler = None
reducer = None


def select_embedding(sc_dataframe, fillup_unmatched=True):
    available_embeddings = os.listdir(PATH_EMBEDDINGS)
    available_embeddings_truncated = [
        x.split('.')[0] for x in available_embeddings]
    if len(available_embeddings_truncated) == 0:
        name_new = input(
            "No embeddings found. Generate new embedding.\nNew embedding name: ")

        if '.' in name_new:
            name_new = name_new.split('.')[0] + '.pkl'
    else:
        str_preview = "Found embeddings:\n"
        for emb in available_embeddings_truncated:
            str_preview += emb + "\n"
        str_preview += "\n"
        name_new = input(
            str_preview +
            "Enter desired embedding, or enter different name to generate new one: ")

    if name_new in available_embeddings_truncated:
        return load_embedding(
            sc_dataframe,
            os.path.join(
                PATH_EMBEDDINGS,
                name_new + '.pkl'),
            fillup_unmatched=fillup_unmatched)
    else:
        return generate_embedding(
            sc_dataframe, os.path.join(
                PATH_EMBEDDINGS, name_new + '.pkl'))


def generate_embedding(sc_dataframe, path_target="", save=True):
    global scaler, reducer

    # create scalers and reducer
    print("\nCalculating embedding")
    sc_dataframe_embedding = sc_dataframe
    cell_data = sc_dataframe_embedding[FEATURES].values
    umap_scaler = StandardScaler().fit(cell_data)
    scaled_cell_data = umap_scaler.transform(cell_data)
    umap_reducer = umap.UMAP(verbose=False).fit(scaled_cell_data)
    embedding = umap_reducer.transform(scaled_cell_data)

    sc_dataframe['x'] = embedding[..., 0]
    sc_dataframe['y'] = embedding[..., 1]

    embedding_data = sc_dataframe[['im_path', 'x', 'y']].copy()
    pickle_storage_object = (embedding_data, umap_scaler, umap_reducer)
    if(save):
        f_obj = open(path_target, 'wb')
        pkl.dump(pickle_storage_object, f_obj)
        f_obj.close()

    scaler, reducer = umap_scaler, umap_reducer
    return sc_dataframe


def load_embedding(sc_dataframe, path, fillup_unmatched):
    global scaler, reducer

    (embedding_data, umap_scaler, umap_reducer) = pkl.load(open(path, 'rb'))

    '''join sc_dataframe by using im_path as index column, and thus load
    columns: x, y and image_column'''
    if len(embedding_data) != len(sc_dataframe):
        print("Mismatch of previously embedded and current cell count. Matching as many cells as possible.")
    elif (embedding_data.im_path != sc_dataframe.im_path).any():
        print("Different dataframe to add embedding to, than which was initially saved. Trying to make the best of it...")

    sc_dataframe['ID'] = sc_dataframe.index
    sc_dataframe = sc_dataframe.set_index('im_path')
    embedding_data = embedding_data.set_index('im_path')
    sc_dataframe = sc_dataframe.drop(['x', 'y'], axis=1, errors='ignore')
    sc_dataframe = sc_dataframe.join(embedding_data, how='left')
    sc_dataframe['im_path'] = sc_dataframe.index
    sc_dataframe = sc_dataframe.set_index('ID')

    scaler, reducer = umap_scaler, umap_reducer

    if(fillup_unmatched):
        idx_unmatched = sc_dataframe['x'].isna()
        df_unmatched = sc_dataframe.loc[idx_unmatched]
        coordinates = embed_new_data(df_unmatched)[['x', 'y']]
        sc_dataframe.loc[idx_unmatched, 'x'] = coordinates['x']
        sc_dataframe.loc[idx_unmatched, 'y'] = coordinates['y']

    return sc_dataframe


def embed_new_data(df):
    if scaler is None:
        raise NameError("No embedding selected!")

    df = df.copy()
    cell_data = df[FEATURES].values
    scaled_cell_data = scaler.transform(cell_data)
    embedding = reducer.transform(scaled_cell_data)
    df['x'] = embedding[..., 0]
    df['y'] = embedding[..., 1]

    return df
