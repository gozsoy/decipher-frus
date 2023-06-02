import os
import pyLDAvis
import spacy
import pickle
import math
import ray
import numpy as np
import pandas as pd
import tomotopy as tp
import constants

nlp = spacy.load('en_core_web_sm')

tables_path, plots_path = constants.TABLES_PATH, constants.PLOTS_PATH
topic_count = constants.LDA_TOPIC_COUNT
name_extension = constants.LDA_NAME_EXTENSION

if not os.path.exists(tables_path):
    os.makedirs(tables_path)

if not os.path.exists(plots_path):
    os.makedirs(plots_path)


# helper function for removing named entities, stopwords, punctuation,
#  words with length smaller than 3 from text
@ray.remote
def preprocess(row):
    # trimming extra long texts for memory in character cnt
    if len(row) > 1000000:
        row = row[:1000000]

    return [t.lemma_.lower() for t in nlp(row) if not t.is_punct 
            and not t.is_stop and t.ent_iob_ == 'O' and len(t.text) >= 3]  
# and t.pos_ in ['PROPN','NOUN']


if __name__ == "__main__":
        
    #####
    # STEP 0: load documents, remove entities, and save
    #####

    doc_df = pd.read_parquet(tables_path+'doc.parquet')

    # if already extracted, use them
    if os.path.isfile(tables_path+"free_text_list"+name_extension):
        with open(tables_path+"free_text_list"+name_extension, "rb") as fp:
            processed_free_text_list = pickle.load(fp)
        print('lda free text list loaded.')

    else:
        free_text_list = doc_df['text'].values

        # check if text None, if so replace it with ' '.
        free_text_list = list(map(lambda x: ' ' if not x or
                              (isinstance(x, float) and math.isnan(x)) else x,
                                free_text_list))

        ray.init(num_cpus=13)
        futures = [preprocess.remote(row) for row in free_text_list]
        processed_free_text_list = ray.get(futures)
        ray.shutdown()

        with open(tables_path+"free_text_list"+name_extension, "wb") as fp:
            pickle.dump(processed_free_text_list, fp)
        print('lda free text list computed and saved.')

    #####
    # STEP 1: fit topic model
    #####

    # define model
    mdl = tp.LDAModel(k=topic_count, tw=tp.TermWeight.ONE)

    # add documents
    for txt in processed_free_text_list:
        mdl.add_doc(txt)

    # train model
    mdl.train(100)

    # save into file
    mdl.save(tables_path+'topic_model'+name_extension+'.bin')

    #####
    # STEP 2: visualization
    #####

    topic_term_dists = np.stack([mdl.get_topic_word_dist(k) 
                                 for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
    vocab = list(mdl.used_vocabs)
    term_frequency = mdl.used_vocab_freq

    prepared_data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, 
                                     doc_lengths, vocab, term_frequency,
                                     start_index=0, sort_topics=False)
    pyLDAvis.save_html(prepared_data, 
                       plots_path+'topics'+name_extension+'.html')

    #####
    # STEP 3: integrate topic model with frus kg
    #####

    lda_topic_descp = {}  # topic: top n words

    for idx in range(mdl.k):
        results = mdl.get_topic_words(idx)
        topic_words = list(map(lambda x: x[0], results))
        lda_topic_descp[str(idx)] = " - ".join(topic_words)

    doc_topic_dict = {}  # id_to_text : assigned_topic

    id_to_text_list = doc_df['id_to_text'].values
    doc_insts = mdl.docs

    cnt = 0
    for idx in range(len(processed_free_text_list)):

        temp_txt = processed_free_text_list[idx]
        if len(temp_txt) != 0:
            results = doc_insts[cnt].get_topics()
            assigned_topic = str(results[0][0])
            doc_topic_dict[id_to_text_list[idx]] = assigned_topic
            cnt += 1

    topic_desc_df = pd.DataFrame(lda_topic_descp.items(), 
                                 columns=['Name', 'Top_n_words'])
    doc_topic_df = pd.DataFrame(doc_topic_dict.items(), 
                                columns=['id_to_text', 'assigned_topic'])

    topic_desc_df.to_parquet(
        tables_path+'topic_descp'+name_extension+'.parquet')
    doc_topic_df.to_parquet(
        tables_path+'doc_topic'+name_extension+'.parquet')
    
    print('finished.')