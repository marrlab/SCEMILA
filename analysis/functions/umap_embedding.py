import umap
import pickle as pkl
import os

features = range(12800)
path_embeddings = 'suppl_data/embeddings'
path_loaded_frames = ''

def select_embedding():
    available_embeddings = os.listdir(path_embeddings)
    available_embeddings_truncated = [x.split('.')[0] for x in available_embeddings]
    if len(available_embeddings_truncated) == 0:
        print("No embeddings found. Generate new embedding.")
        name_new = input("New embedding name: ")
    else:
        print("Found embeddings: ")
        for emb in available_embeddings_truncated:
            print(emb)
        print("")
        name_new = input("Choose embedding, or enter different name to generate new one: ")

    if name_new in available_embeddings_truncated:
        

def generate_embedding(new_name):
    return

def load_embedding(path):
    return
