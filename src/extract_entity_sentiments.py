import spacy
import math
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob
from multiprocessing import Pool

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

tables_path = '../tables/tables_1952_1988/'

# these entities will be omitted when found
unwanted_entities = ['DATE', 'TIME', 'QUANTITY', 'ORDINAL',
                     'CARDINAL', 'MONEY', 'PERCENT']


# helper function that iterates over each sentence in a document
# finds named entities and extracts any adjactives and binds all
def process_document(tpl):
    
    temp_id_to_text, text = tpl
    
    entity_sent_dict = {}

    # check if text is null
    if not (isinstance(text, float) and math.isnan(text)):
        
        # spacy pipeline on raw text
        doc = nlp(text)
        
        # investigate each sentence one by one
        for sent in doc.sents:
            adj_count = 0
            adj_polarity_sum = 0.0
            adj_subjectivity_sum = 0.0

            sentence_named_entities = []
            # (entity,type)
            entity_chunk = None 

            # see if token is part of a named entity
            # if so save ne and also adjective sentiment
            for token in sent:

                if token.ent_iob_ == 'O':
                    if entity_chunk:
                        sentence_named_entities.append(entity_chunk)
                        entity_chunk = None
                    if token.pos_ == 'ADJ' and token._.blob.polarity != 0.0:
                        adj_count += 1
                        adj_polarity_sum += token._.blob.polarity
                        adj_subjectivity_sum += token._.blob.subjectivity
                        
                elif token.ent_iob_ == 'B':
                    if entity_chunk:
                        sentence_named_entities.append(entity_chunk)
                        entity_chunk = None
                    entity_chunk = (token.text, token.ent_type_)
                else:
                    entity_chunk_text = entity_chunk[0]
                    entity_chunk_type = entity_chunk[1]
                    entity_chunk = (entity_chunk_text+' '+token.text, 
                                    entity_chunk_type)
            
            # if adjective count of sentence is more than 0
            # record named entities in sentence in the dict that is returned
            if adj_count > 0:
                sentence_avg_polarity = round(adj_polarity_sum/adj_count, 4)
                sentence_avg_subj = round(adj_subjectivity_sum/adj_count, 4)

                for temp_ne in sentence_named_entities:
                    # if named entity type is valid
                    if temp_ne[1] not in unwanted_entities:
                        if entity_sent_dict.get((temp_ne[0], temp_id_to_text),
                                                None) is None:
                            entity_sent_dict[(temp_ne[0], temp_id_to_text)] \
                                = {'pol': [sentence_avg_polarity], 
                                   'sub': [sentence_avg_subj]}
                        else:
                            entity_sent_dict[(temp_ne[0],
                                              temp_id_to_text)]['pol'].append(
                                sentence_avg_polarity)
                            entity_sent_dict[(temp_ne[0],
                                              temp_id_to_text)]['sub'].append(
                                sentence_avg_subj)

    return entity_sent_dict


if __name__ == '__main__':

    # read files from tables_path
    doc_df = pd.read_parquet(tables_path+'doc.parquet')
    id_to_text_list = doc_df['id_to_text'].values
    free_text_list = doc_df['text'].values

    # parallel processing
    with Pool(13) as p:
        global_entity_sent_list = p.map(process_document,
                                        zip(id_to_text_list,
                                            free_text_list))

    # merge all document results into one big dict
    global_entity_sent_dict = {}
    list(map(lambda x: global_entity_sent_dict.update(x),
         global_entity_sent_list)) 

    # convert dict to pd for better processing
    entity_sent_df = pd.DataFrame(global_entity_sent_dict).transpose()
    entity_sent_df.reset_index(inplace=True)
    entity_sent_df.rename(columns={'level_0': 'named_entity',
                                   'level_1': 'id_to_text'}, inplace=True)
    # save to path
    entity_sent_df.to_parquet(
        tables_path+'entity_sentiment.parquet')

    print('finished.')