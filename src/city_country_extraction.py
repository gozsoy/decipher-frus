import re
# import copy 
# import json
import glob
# import jellyfish
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON
import ray
import os
import constants

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# namespaces for xml parsing
ns = {'xml': 'http://www.w3.org/XML/1998/namespace',
      'dflt': 'http://www.tei-c.org/ns/1.0',
      'frus': 'http://history.state.gov/frus/ns/1.0',
      'xi': 'http://www.w3.org/2001/XInclude'
      }

# variables for wikidata query service
user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
sparqlwd.setReturnFormat(JSON)

tables_path = constants.TABLES_PATH
start_year, end_year = constants.START_YEAR, constants.END_YEAR

if not os.path.exists(tables_path):
    os.makedirs(tables_path)


# helper function 1 step 0
# extract placeName tag within a given document and save it to city_df
@ray.remote
def extract_city(doc):

    # city
    place_tag = doc.find('.//dflt:placeName', ns)
    if place_tag is not None:
        txt = "".join(place_tag.itertext())
        city = " ".join(txt.split())
    else:
        city = None

    return {'placeName': city}


# helper function 1 step 1
# finds a pattern str in a given string str
def geo_match(pattern, string):
    
    if pattern != pattern:
        return None
    elif re.search(pattern, string):
        return pattern
    else:
        return None


# helper function 2 step 1
# searches if extension is either country or subcountry.
# if so return its country
def f(string):

    if not string:
        return None
    
    tl = list(wc_df[wc_df['country'].apply(
        lambda pattern: True if geo_match(pattern, string) else False)]
            .drop_duplicates(subset='country')['country'].values)
    if len(tl) == 0:
        tl = list(wc_df[wc_df['subcountry'].apply(
            lambda pattern: True if geo_match(pattern, string) else False)]
                .drop_duplicates(subset='country')['country'].values)

    if len(tl) == 0:
        return None
    elif len(tl) == 1:
        return tl[0]
    else:
        print(f'multi-match for {string}. Check later!')
        return tl


# helper function 3 step 1
# searches if name is subcountry. if so return its country
def f2(string):

    if not string:
        return None

    tl = list(wc_df[wc_df['subcountry'].apply(
        lambda pattern: True if geo_match(pattern, string) else False)]
        .drop_duplicates(subset='country')['country'].values)

    if len(tl) == 0:
        return None
    elif len(tl) == 1:
        return tl[0]
    else:
        print(f'multi-match for {string}. Check later!')
        return tl


# helper function 1 step 2
# if given string is a capital city in wikidata, fetch its country
def find_if_capital(name):

    try:
        query = """
        SELECT ?country ?countryLabel WHERE {
        SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search  \'"""+name+"""\';
                            mwapi:language "en".
            ?city wikibase:apiOutputItem mwapi:item.
            ?num wikibase:apiOrdinal true.
        }
        ?city wdt:P31 wd:Q5119.
        ?city wdt:P17 ?country.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en".}
        }
        """
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {name}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function 2 step 2
# if given string is a big city in wikidata, fetch its country
def find_if_bigcity(name):

    try:
        query = """
        SELECT ?country ?countryLabel WHERE {
        SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search  \'"""+name+"""\';
                            mwapi:language "en".
            ?city wikibase:apiOutputItem mwapi:item.
            ?num wikibase:apiOrdinal true.
        }
        ?city (wdt:P31/wdt:P279*) wd:Q1549591.
        ?city wdt:P17 ?country.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en".}
        }
        """
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {name}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}


# helper function 3 step 2
# applies one of the above functions to given str, and regulates results
def process_name(row, f):

    name = row['name']

    res = f(name)

    candidates = []
    selected_country = None

    for binding in res['results']['bindings']:
        candidates.append(binding['countryLabel']['value'])

    if len(candidates) > 0:
        temp_country = Counter(candidates).most_common(1)[0][0]
        selected_country = temp_country

    return selected_country


# helper function 4 step 2
# merge two columns about wikidata querying into one
def merger2(row):

    d1 = row['wiki_capital_guess']
    d2 = row['wiki_bigcity_guess']

    if (not d2 or d2 != d2) and (not d1 or d1 != d1):
        return None
    elif (not d2 or d2 != d2):
        return d1
    else:
        return d2


# helper function 5 step 2
# if country field is not filled before by dataset, 
# use info from wikidata to fill it
def merger3(row):

    d1 = row['country']
    d2 = row['merged_wiki']

    if not d2 and d1 != d1:
        return None
    elif d1 != d1:
        return d2
    else:
        return d1


# helper function 6 step 2
# if country field is not filled before by neither dataset nor wikidata, 
# then use wc_guess field
# this means placeName is actually not city but a larger region
def merger4(row):

    d1 = row['country']
    d2 = row['wc_guess']

    if (not d2 or d2 != d2) and (not d1 or d1 != d1):
        return None
    elif (not d1 or d1 != d1):
        return d2
    else:
        return d1


'''# helper function 1 step 3
# computes similarity between two str acc to func
def compute_sim(s1, func, s2):
    return func(s1, s2)


# helper function 2 step 3
# given a city, computes its edit distance againts other cities
# does this for all cities
def find_matches(s2):

    spiro_dist_df = pd.DataFrame(
        {'name_set': all_names, 'dam_lev_dist': 
         [compute_sim(x, jellyfish.damerau_levenshtein_distance, s2)
          for x in all_names]})
    
    misspelling_idx = set(
        spiro_dist_df[(spiro_dist_df['dam_lev_dist'] <= 1)].index.values)

    return misspelling_idx'''


# helper function 1 step 4
# this function is a quick remedy. 
# since number of countries for name fix is limited, can be used all times.
def name_converter(name):

    if name == 'United States':
        return 'United States of America'
    elif name == 'China':
        return "People's Republic of China"
    else:
        return name
    

# helper function 1 step 5
# finds wikidata country entry for given country
def find_country(name):

    try:
        query = """
        SELECT ?country ?countryLabel WHERE {
        SERVICE wikibase:mwapi {
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search  \""""+name+"""\";
                            mwapi:language "en".
            ?country wikibase:apiOutputItem mwapi:item.
            ?num wikibase:apiOrdinal true.
        }
        ?country wdt:P31 wd:Q6256.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en".}
        }
        """
        
        sparqlwd.setQuery(query)

        return sparqlwd.query().convert()

    except Exception as e:
        print(f'name: {name}')
        print(f'error message: {e}')
        return {'head': {'vars': ['item']}, 'results': {'bindings': []}}
    

# helper function 2 step 5
# calls above function and regulates results into our code
def process_name2(name):

    res = find_country(name)

    if len(res['results']['bindings']) > 0:
        binding = res['results']['bindings'][0]

        country = binding['countryLabel']['value']
        tag = binding['country']['value']

        return country, tag
    
    else:
        return name, None


if __name__ == "__main__":

    #####
    # STEP 0: extract placeName tags from each document
    #####
    
    # city_df = pd.DataFrame(columns=['name'])
    ray.init(num_cpus=13)
    global_city_list = []

    for file in tqdm(glob.glob('../volumes/frus*')):
        file_start_year = int(file[15:19])
        
        if file_start_year >= start_year and file_start_year <= end_year:

            tree = ET.parse(file)
            root = tree.getroot()

            docs = root.findall(
                './dflt:text/dflt:body//dflt:div[@type="document"]', ns)
            futures = [extract_city.remote(doc) for doc in docs]
            result_list = ray.get(futures)
            global_city_list += result_list  

    ray.shutdown()
    city_df = pd.DataFrame(global_city_list)
    city_df.dropna(inplace=True)
    city_df.drop_duplicates(inplace=True)
    city_df.reset_index(drop=True, inplace=True)

    # split place names with name-extension pairs if exist
    extension_col = city_df['placeName'].apply(
        lambda x: " ".join(x.split(',')[1:]))
    name_col = city_df['placeName'].apply(lambda x: x.split(',')[0])
    city_df['name'] = name_col
    city_df['extension'] = extension_col
    city_df['extension'] = city_df['extension'].apply(
        lambda x: None if len(x) == 0 else x)

    #####
    # STEP 1: quick matching with a static geonames dataset
    #####

    # using external source for quick matching before wikidata
    wc_df = pd.read_csv('../tables/world-cities.csv')

    city_df['extension_match'] = city_df['extension'].apply(lambda x: f(x))
    city_df['wc_guess'] = city_df[city_df['extension'].isna()]['name']\
        .apply(lambda x: f2(x))
    city_df['country'] = city_df['extension_match']

    # extension match: found country for the extension within static database
    # wc_guess: found country for strings with no extension within database
    city_df = city_df[['placeName', 'name', 'extension', 
                       'country', 'extension_match', 'wc_guess']]
    city_df['placeName'] = city_df['placeName'].apply(lambda x: '\"'+x+'\"')

    # save and edit
    # city_df.to_csv(tables_path+'city_step1.csv')
    print('step 1 done.')

    #####
    # STEP 2: wikidata matching for unmatched cases in step 1 
    # (requires manual work, follow terminal outputs)
    #####

    # load corrected one
    # city_df = pd.read_csv(tables_path+'city_step1.csv')

    # find country if city is capital
    wiki_df = city_df[city_df['extension_match'].apply(
        lambda x: True if x != x else False)]
    city_df['wiki_capital_guess'] = wiki_df.apply(
        process_name, axis=1, f=find_if_capital)

    # find country if city is big city but not capital
    wiki_df = city_df[city_df['extension_match'].apply(
        lambda x: True if x != x else False) & city_df['wiki_capital_guess']
        .apply(lambda x: False if x else True)]
    city_df['wiki_bigcity_guess'] = wiki_df.apply(
        process_name, axis=1, f=find_if_bigcity)

    city_df['merged_wiki'] = city_df.apply(merger2, axis=1)
    city_df['country'] = city_df.apply(merger3, axis=1)
    city_df['country'] = city_df.apply(merger4, axis=1)

    # save and edit
    city_df.to_csv(tables_path+'city_step1.csv')
    input("Go to city_step1.csv. Resolve multi-match cases in 'country' "
          "column by hand. You may correct name-country columns mismatches "
          "by hand as well and press Enter to continue")
    print('step 2 done.')

    #####
    # STEP 3: finalize process with several last steps
    #####

    # load corrected one
    city_df = pd.read_csv(tables_path+'city_step1.csv')
    city_df = city_df[['placeName', 'name', 'extension', 'country']]
    
    # remove double quotes around original name
    city_df['placeName'] = city_df['placeName'].apply(lambda x: x[1:-1])

    # wikidata and static database country name unifier
    city_df['country'] = city_df['country'].apply(name_converter)

    # save as parquet this time
    city_df.to_parquet(tables_path+'city.parquet')

    #####
    # STEP 4: create country-wikiTag dataframe
    #####
    
    city_df = pd.read_parquet(tables_path+'city.parquet')
    country_tag_pairs = list(map(process_name2, city_df['country'].unique()))

    countryLabel = list(map(lambda x: x[0], country_tag_pairs))
    countryTag = list(map(lambda x: x[1], country_tag_pairs))

    country_df = pd.DataFrame.from_dict({'countryLabel': countryLabel, 
                                         'countryTag': countryTag})

    # remove nan entries before saving
    country_df.dropna(inplace=True)

    country_df.to_csv(tables_path+'country.csv')

    print('finished.')


