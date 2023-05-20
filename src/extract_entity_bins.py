import os
import math
import spacy
import numpy as np
import pandas as pd
import ray

nlp = spacy.load('en_core_web_sm')

# define path to save extracted files
tables_path = '../tables/tables_1952_1988/'

# these entities will be omitted when found
uninformative_entities = ['DATE', 'TIME', 'QUANTITY', 'ORDINAL',
                          'CARDINAL', 'MONEY', 'PERCENT' 'PERSON']

# threshold based on count - hyperparameter
min_ne_count = 50

# bin size in years - hyperparameter
bin_size = 4
name_extension = '_'+str(bin_size)+'yearbinned'
bins = list(range(1952, 1990, bin_size))


# helper function that finds each named entity given document text
@ray.remote
def apply_ner(tpl):

    idx, sentence = tpl

    doc_ner_dict_list = []

    id_to_text = id_to_text_list[idx]
    year = year_list[idx]
    era = era_list[idx]    

    # check if text is null
    if not (isinstance(sentence, float) and math.isnan(sentence)):

        # trimming extra long texts for memory in terms of ch
        if len(sentence) > 1000000:
            sentence = sentence[:1000000]

        doc = nlp(sentence)

        named_entities = []

        # (entity,type)
        entity_chunk = None

        for token in doc:
            if token.ent_iob_ == 'O':
                if entity_chunk:
                    named_entities.append(entity_chunk)
                    entity_chunk = None
            elif token.ent_iob_ == 'B':
                if entity_chunk:
                    named_entities.append(entity_chunk)
                    entity_chunk = None
                entity_chunk = (token.text, token.ent_type_)
            else:
                entity_chunk_text = entity_chunk[0]
                entity_chunk_type = entity_chunk[1]
                entity_chunk = (entity_chunk_text+' '+token.text, 
                                entity_chunk_type)

        named_entities = list(filter(
             lambda x: True if x[1] not in uninformative_entities 
             else False, named_entities))
        named_entities = np.unique(named_entities, axis=0) 

        for ne_tuple in named_entities:
            ne = ne_tuple[0]

            doc_ner_dict_list.append({'id_to_text': id_to_text, 
                                      'named_entity': ne, 
                                      'year': int(year), 
                                      'era': era})

    return doc_ner_dict_list


if __name__ == '__main__':

    # load document file
    doc_df = pd.read_parquet(tables_path+'doc.parquet')
    # remove editorial notes since they do not have year
    # doc_df = doc_df[doc_df['subtype'] != 'editorial-note'] now have

    id_to_text_list = doc_df['id_to_text'].values
    free_text_list = doc_df['text'].values
    year_list = list(map(lambda x: str(int(x)), doc_df['year'].values))
    era_list = doc_df['era'].values

    # if entities are already extracted, use them
    if os.path.isfile(tables_path+'entity_unbinned.parquet'):
        ne2doc_df = pd.read_parquet(tables_path+'entity_unbinned.parquet')
        print('ne2doc_df loaded.')

    # if not, iterate over each document parallely to find them
    else:
        global_ner_dict_list = []

        ray.init(num_cpus=13)
        futures = [apply_ner.remote(tpl) for tpl in enumerate(free_text_list)]
        result_list = ray.get(futures)
        global_ner_dict_list += sum(result_list, [])
        ray.shutdown()

        ne2doc_df = pd.DataFrame(global_ner_dict_list)
        ne2doc_df.to_parquet(tables_path+'entity_unbinned.parquet')
        print('ne2doc_df computed and saved.')

    # apply minimum entity count threshold
    ne2doc_df = ne2doc_df.groupby('named_entity').\
        filter(lambda x: len(x) >= min_ne_count)

    # create bins
    labels = []
    for i in range(1, len(bins)):
        labels.append(str(bins[i-1])+'-'+str(bins[i]))

    # bin each year
    ne2doc_df['bin'] = pd.cut(ne2doc_df['year'], bins=bins, 
                              labels=labels, right=True)

    # create entity_name + bin information
    ne2doc_df['dynamic_named_entity'] = ne2doc_df['named_entity'].astype(str)\
        + ' ' + ne2doc_df['bin'].astype(str)

    # store binned named entities
    ne2doc_df.to_parquet(tables_path+'entity'+name_extension+'.parquet')

    print('finished.')