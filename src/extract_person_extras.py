import ssl
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
# import ray
from tqdm import tqdm
import constants
import os
tqdm.pandas()

# necessary setting for wikidata querying
ssl._create_default_https_context = ssl._create_unverified_context
user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
sparqlwd.setReturnFormat(JSON)

tables_path = constants.TABLES_PATH

if not os.path.exists(tables_path):
    os.makedirs(tables_path)


# HELPERS ABOUT PERSON-ITS METADATA
# helper functions for extracting specific person information
def gender_f(Q):
    query = """SELECT ?item ?itemLabel
            WHERE 
            {
            wd:"""+Q+""" wdt:P21 ?item;
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query


def religion_f(Q):
    query = """SELECT ?item ?itemLabel
            WHERE 
            {
            wd:"""+Q+""" wdt:P140 ?item.
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query


def educated_f(Q):
    query = """SELECT ?item ?itemLabel
            WHERE 
            {
            wd:"""+Q+""" wdt:P69 ?item.
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query    


def occupation_f(Q):
    query = """SELECT ?item ?itemLabel
            WHERE 
            {
            wd:"""+Q+""" wdt:P106 ?item.
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query  


def citizenship_f(Q):
    query = """SELECT ?item ?itemLabel ?startyearLabel ?endyearLabel
            WHERE 
            {
            wd:"""+Q+""" p:P27 ?statement1.
            ?statement1 ps:P27 ?item.
            OPTIONAL{?statement1 pq:P580 ?startyear.}
            OPTIONAL{?statement1 pq:P582 ?endyear.}
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query  


def party_f(Q):
    query = """SELECT ?item ?itemLabel ?startyearLabel ?endyearLabel
            WHERE 
            {
            wd:"""+Q+""" p:P102 ?statement1.
            ?statement1 ps:P102 ?item.
            OPTIONAL{?statement1 pq:P580 ?startyear.}
            OPTIONAL{?statement1 pq:P582 ?endyear.}
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query  


def memberof_f(Q):
    query = """SELECT ?item ?itemLabel ?startyearLabel ?endyearLabel
            WHERE 
            {
            wd:"""+Q+""" p:P463 ?statement1.
            ?statement1 ps:P463 ?item.
            OPTIONAL{?statement1 pq:P580 ?startyear.}
            OPTIONAL{?statement1 pq:P582 ?endyear.}
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query  


def positionheld_f(Q):
    query = """SELECT ?item ?itemLabel ?startyearLabel ?endyearLabel
            WHERE 
            {
            wd:"""+Q+""" p:P39 ?statement1.
            ?statement1 ps:P39 ?item.
            OPTIONAL{?statement1 pq:P580 ?startyear.}
            OPTIONAL{?statement1 pq:P582 ?endyear.}
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""
    return query  


# helper function for generic wikidata search
def execute_query(type, entity):

    try:
        sparqlwd.setQuery(function_dict[type](entity))

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {entity}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function for searching all persons wikidata entries from a certain 
# information type and returning in a specific format
def process_query(row, type):
   
    entity = row['selected_wiki_entity']
    
    retrieved = []

    if entity:

        entity = entity.split('/')[-1]

        res = execute_query(type, entity)

        for binding in res['results']['bindings']:
            temp = []
            temp.append(binding['item']['value'])
            temp.append(binding['itemLabel']['value'])
            if binding.get('startyearLabel', None):
                temp.append(binding['startyearLabel']['value'])
            if binding.get('endyearLabel', None):
                temp.append(binding['endyearLabel']['value'])
        
            if len(temp) > 0:
                retrieved.append(temp)

    if len(retrieved) > 0:
        return retrieved
    else:
        return None
    

# HELPERS ABOUT PARTY OR SCHOOL-ITS COUNTRY
# helper function 1 to match entity with its country
# useful for matching political party and school with countries
def get_country_tag(Q):

    try:
        query = """
        SELECT ?country ?countryLabel
        WHERE 
        {
        wd:"""+Q+""" wdt:P17 ?country.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }"""
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {Q}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function 2 to match entity with its country
# regulates wikidata search output into specific format
def process_entities(entity):

    res = get_country_tag(entity.split('/')[-1])
    
    if len(res['results']['bindings']) == 0:
        return ''
    else: 
        # not checking multiple countries since meaningless
        binding = res['results']['bindings'][0]

        return binding['countryLabel']['value']


# helper function 3 to match entity with its country
# searches and adds country info for each unique row (of school or party)
def add_country_info(df):
    party_tag_list = pd.unique(df['info_tag'])
    country_col = list(map(lambda x: process_entities(x), party_tag_list))
    party_tag_country_dict = dict(zip(party_tag_list, country_col))
    return df['info_tag'].apply(lambda x: party_tag_country_dict[x])    


# function nickname:function name dict
function_dict = {'gender': gender_f,
                 'religion': religion_f,
                 'educated': educated_f,
                 'occupation': occupation_f,
                 'positionheld': positionheld_f,
                 'citizenship': citizenship_f,
                 'memberof': memberof_f,
                 'party': party_f}


if __name__ == "__main__":

    # load person data from person_unify.py
    new_unified_person_df = pd.read_parquet(
        tables_path+'unified_person_df_final.parquet')

    # query wikidata independently for each type
    gender_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('gender',))
    print('gender done')
    religion_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('religion',))
    print('religion done')
    educated_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('educated',))
    print('educated done')
    occupation_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('occupation',))
    print('occupation done')
    positionheld_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('positionheld',))
    print('positionheld done')
    citizenship_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('citizenship',))
    print('citizenship done')
    party_series = new_unified_person_df.progress_apply(
        process_query, axis=1, args=('party',))
    print('party done')

    # write gender information within person dataframe and save
    new_unified_person_df['gender'] = list(map(
        lambda x: x[0][1] if x else None, gender_series))
    new_unified_person_df.to_parquet(
        tables_path+'unified_person_df_final.parquet')

    # series save name: series variable name dictionary
    # keys represents how we name corresposding series while saving
    name_series_map = {'religion': religion_series,
                       'school': educated_series,
                       'occupation': occupation_series,
                       'role': positionheld_series,
                       'citizenship': citizenship_series,
                       'political_party': party_series}

    # format series with NO start-end year information into our format
    for series_name in ['religion', 'school', 'occupation']:

        series = name_series_map[series_name]

        temp_df = pd.concat([new_unified_person_df['name_set'], series], 
                            axis=1)
        temp_df.rename(columns={0: 'info_list'}, inplace=True)

        info_df = pd.DataFrame(columns=['name_set', 'info_name', 'info_tag'])

        def aux(row):
            global info_df

            name_set = row['name_set']
            info_list = row['info_list']

            if not info_list:
                info_df = pd.concat((
                    info_df, pd.DataFrame({'name_set': [name_set], 
                                           'info_name': [None],
                                           'info_tag': [None]})))
            else:
                for info in info_list:
                    info_df = pd.concat((
                        info_df, pd.DataFrame({'name_set': [name_set],
                                               'info_name': [info[1]], 
                                               'info_tag': [info[0]]})))
            
            return

        temp_df.apply(lambda x: aux(x), axis=1)

        info_df.dropna(thresh=2, inplace=True)  # exclude persons with no info
        info_df.to_parquet(tables_path+'person_'+series_name+'.parquet')

    # format series with start-end year information into our format
    for series_name in ['role', 'citizenship', 'political_party']:

        series = name_series_map[series_name]

        temp_df = pd.concat([new_unified_person_df['name_set'], series], 
                            axis=1)
        temp_df.rename(columns={0: 'info_list'}, inplace=True)

        info_df = pd.DataFrame(columns=['name_set', 'info_name', 'info_tag', 
                                        'start_year', 'end_year'])
        
        def aux2(row):
            global info_df

            name_set = row['name_set']
            info_list = row['info_list']

            if not info_list:
                info_df = pd.concat((
                    info_df, pd.DataFrame({'name_set': [name_set],
                                           'info_name': [None],
                                           'info_tag': [None],
                                           'start_year': [None],
                                           'end_year': [None]})))
            else:
                for info in info_list:
                    info_df = pd.concat((
                        info_df, pd.DataFrame({'name_set': [name_set],
                                               'info_name': [info[1]],
                                               'info_tag': [info[0]],
                                               'start_year': [info[2] if 
                                                              len(info) > 2 
                                                              else None],
                                               'end_year': [info[3] if 
                                                            len(info) > 3 
                                                            else None]})))
            
            return

        temp_df.apply(lambda x: aux2(x), axis=1)

        info_df.dropna(thresh=2, inplace=True)  # exclude persons with no info
        info_df.to_parquet(tables_path+'person_'+series_name+'.parquet')

    # add country information to political party and school
    person_party_df = pd.read_parquet(
        tables_path+'person_political_party.parquet')
    person_party_df['country'] = add_country_info(person_party_df)
    person_party_df.to_parquet(tables_path+'person_political_party.parquet')
    print('party-country done')

    person_school_df = pd.read_parquet(tables_path+'person_school.parquet')
    person_school_df['country'] = add_country_info(person_school_df)
    person_school_df.to_parquet(tables_path+'person_school.parquet')
    print('school-country done')

    print('finished.')


