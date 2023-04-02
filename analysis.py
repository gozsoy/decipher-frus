import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech, KeyBERTInspired
import sqlite3
import sqllite_handler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import spacy
import pickle
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import os.path
import math

nlp = spacy.load('en_core_web_sm')

USE_EMBEDDINGS = True
USE_KEYBERT = False
name_extension = '_52_88_entremoved'
tables_path = 'tables/tables_52_88/'
plots_path = 'plots/plots_52_88/'

def remove_entities(txt):
    document = nlp(txt)

    edited_txt = ""
    for token in document:
        
        if token.ent_iob_=='O':
            if token.whitespace_:
                edited_txt += token.text+ ' '
            else:
                edited_txt += token.text
    
    return edited_txt


# PRECOMPUTE EMBEDDINGS
if not USE_EMBEDDINGS:

    doc_df = pd.read_csv(tables_path+'doc.csv')
    free_text_list = doc_df['text'].values
    free_text_list = list(map(lambda x: ' ' if not x or (isinstance(x, float) and math.isnan(x)) else x, free_text_list))

    free_text_list = list(map(lambda x: remove_entities(x), free_text_list))
    print('entities removed from free texts.')

    with open(tables_path+"free_text_list"+name_extension, "wb") as fp:
        pickle.dump(free_text_list, fp)

    # prepare embeddings beforehand
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(free_text_list, show_progress_bar=True)
    np.save(tables_path+'embeddings'+name_extension+'.npy',embeddings)
    print('embeddings computed from scratch.')

else:
    doc_df = pd.read_csv(tables_path+'doc.csv')
    with open(tables_path+"free_text_list"+name_extension, "rb") as fp:
        free_text_list = pickle.load(fp)

    free_text_list = list(map(lambda x: ' ' if not x else x, free_text_list))
    embeddings = np.load(tables_path+'embeddings'+name_extension+'.npy')
    print('embeddings loaded.')


#USE KEYBERT DETECTED KEYWORDS
if USE_KEYBERT:
    if os.path.isfile(tables_path+'keybert_vocabulary'):
        with open(tables_path+"keybert_vocabulary", "rb") as fp:
            vocabulary = pickle.load(fp)
        print('keybert keywords loaded.')

    else:
        print('keybert process starting...')
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(free_text_list, use_mmr=True, keyphrase_ngram_range=(1,2), top_n=10)
        print('keywords extracted.')

        vocabulary = [k[0] for keyword in keywords for k in keyword]
        vocabulary = list(set(vocabulary))

        with open(tables_path+"keybert_vocabulary"+name_extension, "wb") as fp:
            pickle.dump(vocabulary, fp)

    vectorizer_model = CountVectorizer(vocabulary=vocabulary)
    
    
else:
    vectorizer_model = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=(1,2))


# FIT TOPIC MODEL
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
mmr = MaximalMarginalRelevance(diversity=0.6)
#pos = PartOfSpeech("en_core_web_sm")
keybert_insp = KeyBERTInspired()

topic_model = BERTopic(verbose=True, embedding_model=sentence_model, vectorizer_model=vectorizer_model, representation_model=[mmr])
print('topic model initialized.')

topics, probs = topic_model.fit_transform(free_text_list,embeddings)
print('topic model fit transform done.')

#topics = topic_model.reduce_outliers(free_text_list, topics)
#print('topics outliers reduced. visualization process starting...')

topic_model.reduce_topics(free_text_list, 'auto')
print('topics reduced.')


# INTEGRATE TOPIC MODEL WITH FRUS KG
messy_doc_topic_df = topic_model.get_document_info(free_text_list)

topic_desc_df = messy_doc_topic_df[['Name','Top_n_words']].drop_duplicates(ignore_index=True)
doc_topic_df = pd.DataFrame({'id_to_text':doc_df['id_to_text'],'assigned_topic':messy_doc_topic_df['Name']})

topic_desc_df.to_csv(tables_path+'topic_descp'+name_extension+'.csv')
doc_topic_df.to_csv(tables_path+'doc_topic'+name_extension+'.csv')
print('topic-kg integration files saved. visualization process starting...')


# VISUALIZATION

# dynamic topic modeling (slow)
'''doc_df = pd.read_csv(tables_path+'doc.csv')
doc_df['year'].fillna(method='bfill',inplace=True) # fill editorial-note dates
timestamps = list(doc_df['year'].apply(lambda x: int(x)))
nr_bins = len(np.unique(timestamps))
topics_over_time = topic_model.topics_over_time(free_text_list, timestamps, nr_bins=nr_bins)

fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=30)
fig.write_html(plots_path+"topics_over_time"+name_extension+".html")'''


fig = topic_model.visualize_barchart(topics=np.unique(topic_model.topics_),n_words=7)
fig.write_html(plots_path+"word_importance_per_topic"+name_extension+".html")
print('plot 1 done.')

fig = topic_model.visualize_documents(docs=free_text_list,embeddings=embeddings)
fig.write_html(plots_path+"umap_document_embeddings"+name_extension+".html")
print('plot 2 done.')

'''
reduced_embeddings = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(embeddings)
fig = topic_model.visualize_documents(docs=free_text_list,reduced_embeddings=reduced_embeddings)
fig.write_html(plots_path+"tsne_document_embeddings"+name_extension+".html")
'''

#fig = topic_model.visualize_topics()
#fig.write_html(plots_path+"topic_embeddings"+name_extension+".html")
#print('plot 3 done.')

fig = topic_model.visualize_heatmap()
fig.write_html(plots_path+"topic_similarity_heatmap"+name_extension+".html")
print('plot 4 done.')

#hierarchical_topics = topic_model.hierarchical_topics(free_text_list)
fig = topic_model.visualize_hierarchy()#hierarchical_topics=hierarchical_topics)
fig.write_html(plots_path+"topic_hierarchy"+name_extension+".html")
print('plot 5 done.')
print('visualization process finished.')


#save model
topic_model.save(tables_path+"topic_model"+name_extension, save_embedding_model=False)
print('topic model saved.')


