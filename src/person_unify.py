import re
import ssl
import copy
import glob
import itertools
import jellyfish
from tqdm import tqdm
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer
import ray
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer, util
tqdm.pandas()

tokenizer = RegexpTokenizer(r'\w+')

# define namespaces in FRUS schema
ns = {'xml': 'http://www.w3.org/XML/1998/namespace',
      'dflt': 'http://www.tei-c.org/ns/1.0',
      'frus': 'http://history.state.gov/frus/ns/1.0',
      'xi': 'http://www.w3.org/2001/XInclude'
      }

# some definitions for wikidata querying
ssl._create_default_https_context = ssl._create_unverified_context
user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
sparqlwd.setReturnFormat(JSON)

# define path to save extracted files
tables_path = 'tables/tables_52_88_demo/'

# only use documents within these years
start_year, end_year = 1952, 1988


# helper function 1 step 0
# parses person item and extracts name, id, description
def extract_person(item, file):
    volume = file[8:-4]

    persName_item = item.find('.//dflt:persName[@xml:id]', ns)

    if persName_item is not None:

        persName_text = "".join(persName_item.itertext())
        person_id = persName_item.attrib[
            '{http://www.w3.org/XML/1998/namespace}id']

        all_text = "".join(item.itertext())
        end_idx = all_text.find(persName_text) + len(persName_text+',')
        person_descp = " ".join(all_text[end_idx:].split())

        person_name = " ".join(
            re.sub(',', '', " ".join(persName_text.split(', ')[::-1])).split())

        person_id = volume + '_' + person_id

    return {'id': person_id, 'name': person_name, 'description': person_descp}


# helper function 1 step 1
def aux(row):
    global unified_person_dict

    if row['name'] in unified_person_dict:
    
        temp_dict = unified_person_dict[row['name']]

        temp_dict['id_list'].append(row['id'])
        temp_dict['description_list'].append(row['description'])
        
    else:
        unified_person_dict[row['name']] = {'id_list': [row['id']],
                                            'description_list': 
                                            [row['description']]}

    return


# helper function 1 step 2
def aux2(row):
    global new_unified_person_dict

    if row['name_set'] in new_unified_person_dict:
    
        temp_dict = new_unified_person_dict[row['name_set']]

        temp_dict['name_list'].append(row['name'])
        temp_dict['id_list'] += row['id_list']
        temp_dict['description_list'] += row['description_list']
    
    else:
        new_unified_person_dict[row['name_set']] = \
            {'name_list': [row['name']], 'id_list': row['id_list'], 
                'description_list': row['description_list']}

    return


# helper function 1 step 3
# computes similarity between two str acc to func
def compute_sim(s1, func, s2):
    return func(s1, s2)


# helper function 2 step 3
# given two strings, tokenizes them, and finds exact token overlap
# for tokens with length at least 3
def compute_exact_word_overlap(s1, s2):
    l1 = set([x for x in list(set(tokenizer.tokenize(s1))) if len(x) >= 3])
    l2 = set([x for x in list(set(tokenizer.tokenize(s2))) if len(x) >= 3])

    return len(l1.intersection(l2))


# helper function 3 step 3
# for a given name computes its string similarity to all other names
# does this for all names separetely
@ray.remote
def find_matches(idx):
    s2 = all_names[idx]
    
    spiro_dist_df = pd.DataFrame({
        'name_set': all_names,
        'overlap_cnt': 
        [compute_exact_word_overlap(x, s2) for x in all_names],
        'dam_lev_dist': 
        [compute_sim(x, jellyfish.damerau_levenshtein_distance, s2) 
            for x in all_names],
        'jaro_sim': 
        [compute_sim(x, jellyfish.jaro_winkler_similarity, s2) 
            for x in all_names]})
    
    # misspelling check - hyperparameter
    misspelling_idx = set(spiro_dist_df[(
        spiro_dist_df['dam_lev_dist'] <= 1)].index.values)

    # near-duplication check - hyperparameter
    spiro_dist_df = spiro_dist_df[spiro_dist_df['overlap_cnt'] >= 2]
    match_idx = set(
        spiro_dist_df[(spiro_dist_df['jaro_sim'] >= 0.9) |
                      (spiro_dist_df['dam_lev_dist'] <= 5)].index.values)

    return match_idx.union(misspelling_idx)


# helper function 1 step 4
# finds entry for person whose name exact matches
def find_wiki_entity(name):

    try:
        query = """
        SELECT ?item WHERE {
        SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search  \'"""+name+"""\';
                            mwapi:language "en".
            ?item wikibase:apiOutputItem mwapi:item.
            ?num wikibase:apiOrdinal true.
        }
        ?item wdt:P31 wd:Q5
        }
        """
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {name}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function 2 step 4
# for a unified person's each name, searches wikidata
# and regulates wikidata output to our code
@ray.remote
def process_name_list(name_list):
    
    ssl._create_default_https_context = ssl._create_unverified_context

    wiki_tag = set()

    for name in name_list:
        res = find_wiki_entity(name)

        for binding in res['results']['bindings']:
            wiki_tag.add(binding['item']['value'])

    return list(wiki_tag)


# helper function 1 step 5
# returns wikidata description of given entity
def get_entity_descp(Q):

    try:
        query = """
        SELECT ?descp
        WHERE 
        {
        wd:"""+Q+""" schema:description ?descp.
        FILTER ( lang(?descp) = "en" )
        }"""
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {Q}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function 2 step 5
# finds entity descriptions for each person and stores
def process_candidate_entities(row):

    q_list = row['wiki_col']
    
    wiki_descp = []

    for q in q_list:
        
        res = get_entity_descp(q.split('/')[-1])
        
        if len(res['results']['bindings']) == 0:
            wiki_descp.append('')
        else:      
            for binding in res['results']['bindings']:

                wiki_descp.append(binding['descp']['value'])

    return wiki_descp


# helper function 3 step 5
# for a given person, finds the most similar wikidata entry description to
# its frus description via cosine similarity on sbert embeddings, and
# assigns that entry as matched entry to that person
@ray.remote
def process_wiki_col(row):
    ssl._create_default_https_context = ssl._create_unverified_context

    wiki_col = row['wiki_col']
    
    if len(wiki_col) == 0:
        return None

    elif len(wiki_col) == 1:
        return wiki_col[0]

    else:
        # compute frus embeddings
        desc_list = row['description_list']
        frus_embedding = np.mean(model.encode(desc_list), axis=0)

        # compute candidate wikidata embeddings
        wiki_descs = process_candidate_entities(row)
        wiki_embeddings = model.encode(wiki_descs)

        # compute cosine sim
        cos_sim = util.cos_sim(frus_embedding, wiki_embeddings)

        # select the highest cosine sim as candidate
        selected_idx = np.argmax(cos_sim, axis=1)[0]
        
        return row["wiki_col"][selected_idx]


if __name__ == "__main__":

    global_person_list = []

    no_annotation_cnt = 0

    #####
    # STEP 0: extract person annotations from each volume
    #####
    for file in tqdm(glob.glob('volumes/frus*')):
        file_start_year = int(file[12:16])
        
        # within confined period
        if file_start_year >= start_year and file_start_year <= end_year:

            tree = ET.parse(file)
            root = tree.getroot()
            persons_section = root.find(
                "./dflt:text/dflt:front//dflt:div[@xml:id='persons']", ns)
            
            # if person annotation exists for this volume
            if persons_section:
                # two different xml flavors exist, both considered here
                for item in persons_section.findall(
                     './/dflt:item/dflt:hi/dflt:persName[@xml:id]/../..', ns):
                    person_dict = extract_person(item, file)
                    global_person_list.append(person_dict) 
                for item in persons_section.findall(
                     './/dflt:item/dflt:persName[@xml:id]/..', ns):
                    person_dict = extract_person(item, file)
                    global_person_list.append(person_dict) 
            else:
                print(f'No person annotation in {file}.')
                no_annotation_cnt += 1

    person_df = pd.DataFrame(global_person_list)
    print(f'Not annotated volume count: {no_annotation_cnt}')
    print(f'Row count: {len(person_df)}')
    print('Step 0 finished.')

    #####
    # STEP 1: reduce exactly matched names
    #####

    unified_person_dict = {}
    person_df.apply(lambda x: aux(x), axis=1)
    unified_person_df = pd.DataFrame.from_dict(
        unified_person_dict, orient='index').reset_index(drop=False)
    unified_person_df.rename(columns={'index': 'name'}, inplace=True)
    print(f'Row count: {len(unified_person_df)}')
    print('Step 1 finished.')
    
    #####
    # STEP 2: reduce names with exactly same words but different combinations
    #####

    unified_person_df['name_set'] = unified_person_df.name.apply(
        lambda x: " ".join(sorted(x.split())))

    new_unified_person_dict = {}

    unified_person_df.apply(lambda x: aux2(x), axis=1)

    new_unified_person_df = pd.DataFrame.from_dict(
        new_unified_person_dict, orient='index').reset_index(drop=False)
    new_unified_person_df.rename(columns={'index': 'name_set'}, inplace=True)
    print(f'Row count: {len(new_unified_person_df)}')
    print('Step 2 finished.')

    #####
    # STEP 3: find and reduce near-duplicate names + obvious misspellings
    #####

    all_names = new_unified_person_df['name_set'].values

    # parallel processing init
    ray.init(num_cpus=13)
    
    # find near-duplicates and typos for each name againts others
    futures = [find_matches.remote(idx) for idx in range(len(all_names))]
    result_tuple_list = ray.get(futures)
    ray.shutdown()

    # name : matched names dict
    t = {}
    for idx in range(len(all_names)):
        t[idx] = result_tuple_list[idx]

    # code to merge found matches
    # finds friend of friend is friend!
    scratch_t = copy.deepcopy(t)
    changed_flag = True

    while changed_flag:

        changed_flag = False

        for key in t:
            
            for matched_idx in t[key]:

                if key != matched_idx:
                    if scratch_t.get(key, None) and \
                     scratch_t.get(matched_idx, None):
                        changed_flag = True
                        t[key] = t[key].union(t[matched_idx])
                        scratch_t.pop(matched_idx, None)
            
        unwanted = set(t.keys()) - set(scratch_t.keys())
        print(f'removing {len(unwanted)} keys.')
        for unwanted_key in unwanted:
            del t[unwanted_key]
        scratch_t = copy.deepcopy(t)
        print('---')
        
    # reduce matched names into single entry
    for temp_key in t:
        
        te_df = new_unified_person_df.iloc[list(t[temp_key])]

        name_list = list(itertools.chain.from_iterable(
            te_df['name_list'].values))
        id_list = list(itertools.chain.from_iterable(
            te_df['id_list'].values))
        description_list = list(itertools.chain.from_iterable(
            te_df['description_list'].values))

        new_unified_person_df.at[temp_key, 'name_list'] = name_list
        new_unified_person_df.at[temp_key, 'id_list'] = id_list
        new_unified_person_df.at[temp_key, 'description_list'] = \
            description_list

    new_unified_person_df = new_unified_person_df.loc[t.keys()]

    # save unified person table
    new_unified_person_df.to_parquet(
        tables_path+'unified_person_df_step3.parquet')
    print(f'Row count: {len(new_unified_person_df)}')
    print('Step 3 finished.')
    
    #####
    # STEP 4: find each person's wikidata entity
    #####
    
    # init and run parallel processing
    ray.init(num_cpus=13)
    new_unified_person_df = pd.read_parquet(
        tables_path+'unified_person_df_step3.parquet')
    futures = [process_name_list.remote(row) 
               for row in new_unified_person_df['name_list'].values]
    wiki_col = ray.get(futures)
    ray.shutdown()

    new_unified_person_df['wiki_col'] = wiki_col
    new_unified_person_df.to_parquet(
        tables_path+'unified_person_df_step4.parquet')
    print(f'Row count: {len(new_unified_person_df)}')
    print('Step 4 finished.')
    
    #####
    # STEP 5: reduce multiple candidate wikidata entities to single using 
    # sbert for each person, if exists
    #####

    # initialize sentence model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # again multiprocessing
    ray.init(num_cpus=8)
    new_unified_person_df_wikicol = pd.read_parquet(
        tables_path+'unified_person_df_step4.parquet')

    futures = [process_wiki_col.remote(row) 
               for _, row in new_unified_person_df_wikicol.iterrows()]
    selected_wiki_entity = ray.get(futures)
    ray.shutdown()

    new_unified_person_df['selected_wiki_entity'] = selected_wiki_entity
    new_unified_person_df.to_parquet(
        tables_path+'unified_person_df_step5.parquet')
    print(f'Row count: {len(new_unified_person_df)}')
    print('Step 5 finished.')
    
    #####
    # STEP 6: reduce names with exactly same wikidata entries
    #####

    new_unified_person_df = pd.read_parquet(
        tables_path+'unified_person_df_step5.parquet')
    
    # find same entried persons
    t = {}

    for idx, key in new_unified_person_df.iterrows():

        ent = key['selected_wiki_entity']

        if not ent:
            t[idx] = set([idx])
        else:
            t[idx] = set(new_unified_person_df[
                new_unified_person_df['selected_wiki_entity'] == ent].index)

    # code to merge found matches
    # finds friend of friend is friend!
    scratch_t = copy.deepcopy(t)
    changed_flag = True

    while changed_flag:

        changed_flag = False

        for key in t:
            
            for matched_idx in t[key]:

                if key != matched_idx:
                    if scratch_t.get(key, None) and\
                          scratch_t.get(matched_idx, None):
                        changed_flag = True
                        t[key] = t[key].union(t[matched_idx])
                        scratch_t.pop(matched_idx, None)
            
        unwanted = set(t.keys()) - set(scratch_t.keys())
        print(f'removing {len(unwanted)} keys.')
        for unwanted_key in unwanted:
            del t[unwanted_key]
        scratch_t = copy.deepcopy(t)
        print('---')

    for temp_key in t:
        
        te_df = new_unified_person_df.loc[list(t[temp_key])]

        name_list = list(itertools.chain.from_iterable(
            te_df['name_list'].values))
        id_list = list(itertools.chain.from_iterable(
            te_df['id_list'].values))
        description_list = list(itertools.chain.from_iterable(
            te_df['description_list'].values))

        new_unified_person_df.at[temp_key, 'name_list'] = name_list
        new_unified_person_df.at[temp_key, 'id_list'] = id_list
        new_unified_person_df.at[temp_key, 'description_list'] =\
            description_list

    new_unified_person_df = new_unified_person_df.loc[t.keys()]

    new_unified_person_df.to_parquet(
        tables_path+'unified_person_df_final.parquet')
    print(f'Row count: {len(new_unified_person_df)}')
    print('Step 6 finished.')
    print('finished.')

