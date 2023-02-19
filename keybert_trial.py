import pandas as pd
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

doc_df = pd.read_csv('tables/doc_69_76v30.csv')
txt_list = doc_df['text'].values


vectorizer = KeyphraseCountVectorizer()
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(txt_list[0],vectorizer=vectorizer,use_mmr=True)


