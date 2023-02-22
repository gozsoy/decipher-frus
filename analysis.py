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

nlp = spacy.load('en_core_web_sm')

USE_EMBEDDINGS = True
USE_KEYBERT = False

def remove_persons(txt):
    document = nlp(txt)

    edited_txt = ""
    for ent in document:
        
        if ent.ent_type_=='PERSON':
            if ent.whitespace_:
                edited_txt += 'Person'+ ' '
            else:
                edited_txt += 'Person'
        else:
            if ent.whitespace_:
                edited_txt += ent.text+ ' '
            else:
                edited_txt += ent.text
    
    return edited_txt

# PRECOMPUTE EMBEDDINGS
if not USE_EMBEDDINGS:

    conn = sqlite3.connect('tables/texts_69_76.db')
    cur = conn.cursor()

    res = cur.execute("SELECT TEXT FROM transcript")
    fetched = res.fetchall()
    free_text_list = list(map(lambda x: x[0], fetched))
    #free_text_list = list(map(lambda x: remove_persons(x), free_text_list))

    with open("plots/free_text_list", "wb") as fp:
        pickle.dump(free_text_list, fp)
    free_text_list = list(map(lambda x: ' ' if not x else x, free_text_list))

    # prepare embeddings beforehand
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(free_text_list, show_progress_bar=True)
    np.save('plots/embeddings_69_76.npy',embeddings)
    print('embeddings computed from scratch.')

else:
    with open("plots/free_text_list", "rb") as fp:
        free_text_list = pickle.load(fp)

    free_text_list = list(map(lambda x: ' ' if not x else x, free_text_list))
    embeddings = np.load('plots/embeddings_69_76.npy')
    print('embeddings loaded.')


#USE KEYBERT DETECTED KEYWORDS
if USE_KEYBERT:
    if os.path.isfile('plots/keybert_vocabulary'):
        with open("plots/keybert_vocabulary", "rb") as fp:
            vocabulary = pickle.load(fp)
        print('keybert keywords loaded.')

    else:
        print('keybert process starting...')
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(free_text_list, use_mmr=True, keyphrase_ngram_range=(1,2), top_n=10)
        print('keywords extracted.')

        vocabulary = [k[0] for keyword in keywords for k in keyword]
        vocabulary = list(set(vocabulary))

        with open("plots/keybert_vocabulary", "wb") as fp:
            pickle.dump(vocabulary, fp)

    vectorizer_model = CountVectorizer(vocabulary=vocabulary)
    
    
else:
    vectorizer_model = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(['Person','person']), ngram_range=(1,2))


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
print('topics reduced. visualization process starting...')


# VISUALIZATION

# dynamic topic modeling
"""doc_df = pd.read_csv('tables/doc_69_76.csv')
doc_df['year'].fillna(method='bfill',inplace=True) # fill editorial-note dates
timestamps = list(doc_df['year'].apply(lambda x: int(x)))
nr_bins = len(np.unique(timestamps))
topics_over_time = topic_model.topics_over_time(free_text_list, timestamps, nr_bins=nr_bins)

fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=15)
fig.write_html("plots/topics_over_time_69_76.html")"""


fig = topic_model.visualize_barchart(topics=np.unique(topic_model.topics_),n_words=7)
fig.write_html("plots/word_importance_per_topic_69_76.html")

fig = topic_model.visualize_documents(docs=free_text_list,embeddings=embeddings)
fig.write_html("plots/umap_document_embeddings_69_76.html")

'''reduced_embeddings = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(embeddings)
fig = topic_model.visualize_documents(docs=free_text_list,reduced_embeddings=reduced_embeddings)
fig.write_html("plots/tsne_document_embeddings_69_76.html")'''

fig = topic_model.visualize_topics()
fig.write_html("plots/topic_embeddings_69_76.html")

fig = topic_model.visualize_heatmap()
fig.write_html("plots/topic_similarity_heatmap_69_76.html")

#hierarchical_topics = topic_model.hierarchical_topics(free_text_list)
fig = topic_model.visualize_hierarchy()#hierarchical_topics=hierarchical_topics)
fig.write_html("plots/topic_hierarchy_69_76.html")
print('visualization process finished.')


#save model
topic_model.save("plots/topic_model_69_76", save_embedding_model=False)


