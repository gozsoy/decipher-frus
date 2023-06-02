import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import constants
import os
tqdm.pandas()

tables_path = constants.TABLES_PATH

if not os.path.exists(tables_path):
    os.makedirs(tables_path)


# helper function to compute average sentence embedding
# of a person's FRUS descriptions
def get_descp_embd(row):
    
    global descp_embd_list

    description_list = row['description_list']

    frus_embedding = np.mean(model.encode(description_list), axis=0)

    descp_embd_list.append(frus_embedding)

    return


# helper function to find most similar persons according to
# cosine similarity on description embeddings
def compute_most_similar_persons(idx, top_n=10):

    global similar_persons_dict

    similar_entity_idx = np.argsort(cossim_mat[idx])[::-1][1:top_n+1]
    
    current_name_set = unified_person_df_final.iloc[idx].name_set
    similar_name_set = unified_person_df_final.iloc[similar_entity_idx]\
        .name_set.values

    similar_persons_dict[current_name_set] = similar_name_set

    return


if __name__ == "__main__":

    # initialize s-bert
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # load unified person data
    unified_person_df_final = pd.read_parquet(
        tables_path+'unified_person_df_final.parquet')

    descp_embd_list = []

    # compute FRUS description embeddings
    unified_person_df_final.progress_apply(lambda x: get_descp_embd(x), axis=1)

    # find most similar persons
    cossim_mat = cosine_similarity(descp_embd_list)

    similar_persons_dict = {}

    for idx in tqdm(range(len(descp_embd_list))):
        compute_most_similar_persons(idx) 

    # save results
    similar_descp_persons = pd.Series(similar_persons_dict).explode().\
        reset_index().rename(columns={'index': 'source_person',
                                      0: 'target_person'})
    similar_descp_persons.to_parquet(
        tables_path+'similar_descp_persons.parquet')
    
    print('finished.')