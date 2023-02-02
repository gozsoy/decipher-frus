import pandas as pd
import numpy as np
from bertopic import BERTopic
import sqlite3
import sqllite_handler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import spacy
import pickle
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')

USE_EMBEDDINGS = True

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

else:

    with open("plots/free_text_list", "rb") as fp:
        free_text_list = pickle.load(fp)

    free_text_list = list(map(lambda x: ' ' if not x else x, free_text_list))
    embeddings = np.load('plots/embeddings_69_76.npy')

vectorizer_model = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(['Person','person']), ngram_range=(1,2))

topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model, diversity=0.5)

topics, probs = topic_model.fit_transform(free_text_list,embeddings)


fig = topic_model.visualize_barchart(topics=np.unique(topic_model.topics_),n_words=7)
fig.write_html("plots/word_importance_per_topic_69_76.html")

fig = topic_model.visualize_documents(docs=free_text_list,embeddings=embeddings)
fig.write_html("plots/umap_document_embeddings_69_76.html")

reduced_embeddings = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(embeddings)
fig = topic_model.visualize_documents(docs=free_text_list,reduced_embeddings=reduced_embeddings)
fig.write_html("plots/tsne_document_embeddings_69_76.html")

fig = topic_model.visualize_topics()
fig.write_html("plots/topic_embeddings_69_76.html")

fig = topic_model.visualize_heatmap()
fig.write_html("plots/topic_similarity_heatmap_69_76.html")

fig = topic_model.visualize_hierarchy()
fig.write_html("plots/topic_hierarchy_69_76.html")

topic_model.save("plots/topic_model_69_76", save_embedding_model=False)


