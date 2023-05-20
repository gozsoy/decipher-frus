import re
import copy
import glob 
import itertools
import jellyfish
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer
import ray

tokenizer = RegexpTokenizer(r'\w+')

# define namespaces in FRUS schema
ns = {'xml': 'http://www.w3.org/XML/1998/namespace',
      'dflt': 'http://www.tei-c.org/ns/1.0',
      'frus': 'http://history.state.gov/frus/ns/1.0',
      'xi': 'http://www.w3.org/2001/XInclude'
      }

# define path to save extracted files
tables_path = '../tables/tables_1952_1988/'

# only use documents within these years
start_year, end_year = 1952, 1988


# helper function 1 step 0
# parses term item and extracts name, id, description
def extract_institution(item, file):
    volume = file[11:-4]

    term_item = item.find('.//dflt:term[@xml:id]', ns)

    if term_item is not None:

        term_text = "".join(term_item.itertext())
        term_id = term_item.attrib['{http://www.w3.org/XML/1998/namespace}id']

        all_text = "".join(item.itertext())
        end_idx = all_text.find(term_text) + len(term_text+',')
        item_descp = " ".join(all_text[end_idx:].split())

        term_name = " ".join(re.sub(',', '', " ".join(
            term_text.split(', ')[::-1])).split())

        term_id = volume + '_' + term_id

    return {'id': term_id,
            'name': term_name, 
            'description': item_descp.lower()}


# helper function 1 step 1
def aux(row):
    global unified_term_dict

    if row['description'] in unified_term_dict:
      
        temp_dict = unified_term_dict[row['description']]

        temp_dict['id_list'].append(row['id'])
        temp_dict['name_list'].append(row['name'])
    
    else:
        unified_term_dict[row['description']] = {'id_list': [row['id']],
                                                 'name_list': [row['name']]}

    return


# helper function 1 step 2
def aux2(row):
    global new_unified_institution_dict

    if row['description_set'] in new_unified_institution_dict:
      
        temp_dict = new_unified_institution_dict[row['description_set']]

        temp_dict['description_list'].append(row['description'])
        temp_dict['id_list'] += row['id_list']
        temp_dict['name_list'] += row['name_list']
    
    else:
        new_unified_institution_dict[row['description_set']] \
            = {'description_list': [row['description']],
               'id_list': row['id_list'],
               'name_list': row['name_list']}

    return


# helper function 1 step 3
# computes similarity between two str acc to func
def compute_sim(s1, func, s2):
    return func(s1, s2)


# helper function 2 step 3
# for a given name computes its string similarity to all other names
# does this for all names separetely
@ray.remote
def find_matches(idx):
    
    s2 = all_descriptions[idx]

    spiro_dist_df = pd.DataFrame(
        {'description_set': all_descriptions,
         'dam_lev_dist': 
         [compute_sim(x, jellyfish.damerau_levenshtein_distance, s2) 
          for x in all_descriptions],
         'jaro_sim': 
         [compute_sim(x, jellyfish.jaro_winkler_similarity, s2) 
          for x in all_descriptions]})
    
    # misspelling check - hyperparameter
    misspelling_idx = set(
        spiro_dist_df[(spiro_dist_df['dam_lev_dist'] <= 2)].index.values)

    return misspelling_idx


if __name__ == "__main__":

    #####
    # STEP 0: extract term annotations from each volume
    #####

    global_term_list = []

    no_annotation_cnt = 0

    for file in tqdm(glob.glob('../volumes/frus*')):

        file_start_year = int(file[15:19])
        
        # within confined period
        if file_start_year >= start_year and file_start_year <= end_year:

            tree = ET.parse(file)
            root = tree.getroot()
            terms_section = root.find(
                "./dflt:text/dflt:front//dflt:div[@xml:id='terms']", ns)
            
            # if term annotation exists for this volume
            if terms_section:
                # two different xml flavors exist, both considered here
                for item in terms_section.findall(
                     './/dflt:item/dflt:hi/dflt:term[@xml:id]/../..', ns):
                    term_dict = extract_institution(item, file)
                    global_term_list.append(term_dict)
                for item in terms_section.findall(
                     './/dflt:item/dflt:term[@xml:id]/..', ns):
                    term_dict = extract_institution(item, file)
                    global_term_list.append(term_dict)
            else:
                print(f'No term annotation in {file}.')
                no_annotation_cnt += 1

    institution_df = pd.DataFrame(global_term_list)
    print(f'Not annotated volume count: {no_annotation_cnt}')
    print(f'Row count: {len(institution_df)}')
    print('Step 0 finished.')

    #####
    # STEP 1: reduce exactly matched institution descriptions
    #####

    unified_term_dict = {}
    institution_df.apply(lambda x: aux(x), axis=1)
    unified_institution_df = pd.DataFrame.from_dict(
        unified_term_dict, orient='index').reset_index(drop=False)
    unified_institution_df.rename(
        columns={'index': 'description'}, inplace=True)

    print(f'Row count: {len(unified_institution_df)}')
    print('Step 1 finished.')

    #####
    # STEP 2: reduce descriptions with exactly same words but different
    # combinations
    #####

    unified_institution_df['description_set'] = \
        unified_institution_df.description.apply(
        lambda x: " ".join(sorted(x.split())))
    new_unified_institution_dict = {}

    unified_institution_df.apply(lambda x: aux2(x), axis=1)

    new_unified_institution_df = pd.DataFrame.from_dict(
        new_unified_institution_dict, orient='index').reset_index(drop=False)
    new_unified_institution_df.rename(
        columns={'index': 'description_set'}, inplace=True)

    print(f'Row count: {len(new_unified_institution_df)}')
    print('Step 2 finished.')

    #####
    # STEP 3: find and reduce obvious misspellings
    #####

    all_descriptions = new_unified_institution_df['description_set'].values

    # find typos for each name againts others
    ray.init(num_cpus=13)
    futures = [find_matches.remote(idx) 
               for idx in range(len(all_descriptions))]
    result_tuple_list = ray.get(futures)
    ray.shutdown()

    # name : matched names dict
    t = {}
    for idx in range(len(all_descriptions)):
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
        
        te_df = new_unified_institution_df.iloc[list(t[temp_key])]

        name_list = list(itertools.chain.from_iterable(
            te_df['name_list'].values))
        id_list = list(itertools.chain.from_iterable(
            te_df['id_list'].values))
        description_list = list(itertools.chain.from_iterable(
            te_df['description_list'].values))

        new_unified_institution_df.at[temp_key, 'name_list'] = name_list
        new_unified_institution_df.at[temp_key, 'id_list'] = id_list
        new_unified_institution_df.at[temp_key, 'description_list'] = \
            description_list

    new_unified_institution_df = new_unified_institution_df.loc[t.keys()]

    # save unified term table
    new_unified_institution_df.to_parquet(
        tables_path+'unified_term_df.parquet')

    print(f'Row count: {len(new_unified_institution_df)}')
    print('Step 3 finished.')

    print('finished.')


