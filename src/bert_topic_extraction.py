import math
import pickle
import spacy
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sentence_transformers import SentenceTransformer
import ray


nlp = spacy.load('en_core_web_sm')

tables_path = '../tables/tables_1952_1988/'
plots_path = '../plots/plots_1952_1988/'
name_extension = '_entremoved'  # or '' for original docs

# If document embeddings already calculated, set True. if not, set False.
# Be aware! Sbert runs over each document, it takes time without GPU.
# We suggest using GPU and storing in tables_path beforehand.
USE_EMBEDDINGS = True

# Remove named entities from text. only valid if USE_EMBEDDINGS=False.
REMOVE_ENTITIES = True


# helper function for removing named entities from text
@ray.remote
def remove_entities(txt):

    # trimming extra long texts for memory in character cnt
    if len(txt) > 1000000:
        txt = txt[:1000000]

    document = nlp(txt)

    edited_txt = ""
    for token in document:
        
        if token.ent_iob_ == 'O':
            if token.whitespace_:
                edited_txt += token.text + ' '
            else:
                edited_txt += token.text
    
    return edited_txt


if __name__ == "__main__":

    #####
    # STEP 0: load documents, compute their embeddings, and save both
    #####
    if not USE_EMBEDDINGS:

        doc_df = pd.read_parquet(tables_path+'doc.parquet')
        free_text_list = doc_df['text'].values

        # check if text None, if so replace it with ' '.
        free_text_list = list(map(lambda x: ' ' if not x or 
                              (isinstance(x, float) and math.isnan(x)) else x,
                                free_text_list))
        
        # remove named entities
        if REMOVE_ENTITIES:
            ray.init(num_cpus=13)
            futures = [remove_entities.remote(txt) for txt in free_text_list]
            free_text_list = ray.get(futures)
            ray.shutdown()
            print('entities removed from free texts.')

        # save free texts in tables_path
        with open(tables_path+"free_text_list"+name_extension, "wb") as fp:
            pickle.dump(free_text_list, fp)

        # compute sentence embeddings
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(free_text_list, 
                                           show_progress_bar=True)
        np.save(tables_path+'embeddings'+name_extension+'.npy', embeddings)
        print('embeddings computed from scratch.')

    # load saved documents and embeddings
    else:
        doc_df = pd.read_parquet(tables_path+'doc.parquet')
        with open(tables_path+"free_text_list"+name_extension, "rb") as fp:
            free_text_list = pickle.load(fp)

        free_text_list = list(map(lambda x: ' ' if not x 
                                  else x, free_text_list))
        embeddings = np.load(tables_path+'embeddings'+name_extension+'.npy')
        print('embeddings loaded.')

    #####
    # STEP 1: fit topic model
    #####
    vectorizer_model = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS,
                                       ngram_range=(1, 2))
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    mmr = MaximalMarginalRelevance(diversity=0.6)
    # other possible representation models
    # pos = PartOfSpeech("en_core_web_sm")
    # keybert_insp = KeyBERTInspired()

    topic_model = BERTopic(embedding_model=sentence_model, 
                           vectorizer_model=vectorizer_model,
                           representation_model=[mmr], 
                           verbose=True)
    print('topic model initialized.')

    topics, probs = topic_model.fit_transform(free_text_list, embeddings)
    print('topic model fit transform done.')

    # topics = topic_model.reduce_outliers(free_text_list, topics)
    # print('topics outliers reduced. visualization process starting...')

    topic_model.reduce_topics(free_text_list, nr_topics=100)
    print('topics reduced.')

    #####
    # STEP 2: integrate topic model with frus kg
    #####

    # get document, topic, and descriptive words
    messy_doc_topic_df = topic_model.get_document_info(free_text_list)

    # save topics and their descriptive words
    topic_desc_df = messy_doc_topic_df[['Name', 'Top_n_words']].\
        drop_duplicates(ignore_index=True)
    # save documents and their topic assignment
    doc_topic_df = pd.DataFrame({'id_to_text': doc_df['id_to_text'],
                                 'assigned_topic': messy_doc_topic_df['Name']})

    topic_desc_df.to_csv(tables_path+'topic_descp'+name_extension+'.csv')
    doc_topic_df.to_csv(tables_path+'doc_topic'+name_extension+'.csv')
    print('topic-kg integration files saved. visualization starting...')

    #####
    # STEP 3: visualization
    #####

    fig = topic_model.visualize_barchart(
        topics=np.unique(topic_model.topics_), n_words=7)
    fig.write_html(plots_path+"word_importance_per_topic" +
                   name_extension+".html")
    print('plot 1 done.')

    '''fig = topic_model.visualize_documents(
        docs=free_text_list, embeddings=embeddings)
    fig.write_html(plots_path+"umap_document_embeddings" + 
                   name_extension+".html")
    print('plot 2 done.')'''

    fig = topic_model.visualize_heatmap()
    fig.write_html(plots_path+"topic_similarity_heatmap" + 
                   name_extension+".html")
    print('plot 3 done.')

    hierarchical_topics = topic_model.hierarchical_topics(free_text_list)
    fig = topic_model.visualize_hierarchy(
        hierarchical_topics=hierarchical_topics)
    fig.write_html(plots_path+"topic_hierarchy"+name_extension+".html")
    print('plot 4 done.')
    
    # dynamic topic modeling. Beware: slow!
    '''
    doc_df = pd.read_csv(tables_path+'doc.csv')
    # fill editorial-note dates
    doc_df['year'].fillna(method='bfill',inplace=True)
    timestamps = list(doc_df['year'].apply(lambda x: int(x)))
    nr_bins = len(np.unique(timestamps))
    topics_over_time = topic_model.topics_over_time(
        free_text_list, timestamps, nr_bins=nr_bins)
    fig = topic_model.visualize_topics_over_time(
        topics_over_time, top_n_topics=30)
    fig.write_html(plots_path+"topics_over_time"+name_extension+".html")
    print('plot 5 done.')
    '''
    print('visualization process finished.')

    # save topic model
    topic_model.save(tables_path+"topic_model"+name_extension, 
                     save_embedding_model=False)
    print('topic model saved.')
    
    print('finished.')

