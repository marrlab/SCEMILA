import umap
import pickle as pkl
import os
import image_bytestream
from sklearn.preprocessing import StandardScaler

features = range(12800)
path_embeddings = 'suppl_data/embeddings'
path_loaded_frames = ''

def select_embedding(sc_dataframe):
    available_embeddings = os.listdir(path_embeddings)
    available_embeddings_truncated = [x.split('.')[0] for x in available_embeddings]
    if len(available_embeddings_truncated) == 0:
        name_new = input("No embeddings found. Generate new embedding.\nNew embedding name: ")
        
        if '.' in name_new:
            name_new = name_new.split('.')[0] + '.pkl'
    else:
        str_preview = "Found embeddings:\n"
        for emb in available_embeddings_truncated:
            str_preview += emb + "\n"
        str_preview += "\n"
        name_new = input(str_preview + "Enter desired embedding, or enter different name to generate new one: ")

    if name_new in available_embeddings_truncated:
        return load_embedding(sc_dataframe, os.path.join(path_embeddings, name_new +'.pkl'))
    else:
        return generate_embedding(sc_dataframe, os.path.join(path_embeddings, name_new + '.pkl'))

def generate_embedding(sc_dataframe, path_target, save=False):
    
    # create scalers and reducer
    print("\nCalculating embedding")
    sc_dataframe_embedding = sc_dataframe
    cell_data = sc_dataframe_embedding[features].values
    umap_scaler = StandardScaler().fit(cell_data)
    scaled_cell_data = umap_scaler.transform(cell_data)
    umap_reducer = umap.UMAP(verbose=True)
    embedding = umap_reducer.fit_transform(scaled_cell_data)
    
    
    sc_dataframe['x'] = embedding[..., 0]
    sc_dataframe['y'] = embedding[..., 1]
    
    embedding_data = sc_dataframe[['im_path', 'x', 'y']].copy()
    pickle_storage_object = (umap_scaler, umap_reducer, embedding_data)
    if(save):
        f_obj = open(path_target, 'wb')
        pkl.dump(pickle_storage_object, f_obj)
        f_obj.close()
    
    return sc_dataframe

def load_embedding(sc_dataframe, path):
    umap_scaler, umap_reducer, embedding_data = pkl.load(open(path, 'rb'))
    
    '''join sc_dataframe by using im_path as index column, and thus load 
    columns: x, y and image_column'''
    if len(embedding_data) != len(sc_dataframe):
        print("Mismatch of previously embedded and current cell count. Scaling...")
        sc_dataframe_scaled = umap_scaler.transform(sc_dataframe[features].values)
        print("Calculating coordinates. This will take some time ...")
        embedding = umap_reducer.transform(sc_dataframe_scaled)
    
        sc_dataframe['x'] = embedding[..., 0]
        sc_dataframe['y'] = embedding[..., 1]
    else:
        sc_dataframe['ID'] = sc_dataframe.index
        sc_dataframe = sc_dataframe.set_index('im_path')
        embedding_data = embedding_data.set_index('im_path')
        sc_dataframe = sc_dataframe.drop(['x','y'], axis=1, errors='ignore')
        sc_dataframe = sc_dataframe.join(embedding_data).set_index('ID')
    return sc_dataframe