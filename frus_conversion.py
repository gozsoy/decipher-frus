
# Hack to make the module importable
import sys
sys.path.append(r'./../')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from py2neo import Graph, NodeMatcher
import pandas as pd

from rel2graph.relational_modules.pandas import PandasDataframeIterator
from rel2graph import IteratorIterator
from rel2graph import Converter
from rel2graph.utils import load_file
from rel2graph import register_attribute_postprocessor, Attribute, register_attribute_preprocessor, Resource, register_subgraph_preprocessor
import rel2graph.common_modules
from rel2graph.common_modules import DATE
from datetime import datetime
import numpy as np
import math

filename = "frus_schema.yaml"


# Configure Logging
import logging

logging.basicConfig(filename='tables/rel2graphlogs.log',level=logging.WARNING)
logger = logging.getLogger("rel2graph")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("%(asctime)s [%(threadName)s]::[%(levelname)s]::%(filename)s: %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


import sqlite3
import sqllite_handler

rdb_name = 'tables/texts_69_76.db'

# Setup sqlite database for transcript texts
# Create table if not exists
print("Setting up sqlite database.")
conn = sqlite3.connect(rdb_name)

conn.execute('''CREATE TABLE IF NOT EXISTS transcript
        (ID PRIMARY KEY NOT NULL,
        TEXT);''')
conn.close()

sqllite_handler.init(rdb_name)


#doc_df = pd.read_csv('tables/doc_69_76.csv') VALID ONE
doc_df = pd.read_csv('tables/doc_69_76v30.csv') # EXPERIMENTAL PURPOSES

# change year from type 'float' to 'str(int)' suitable for rel2graph
doc_df['year'] = doc_df['year'].apply(lambda x: x if math.isnan(x) else str(int(x)))

country_df = pd.read_csv('tables/country_69_76.csv')
city_country_df = pd.read_parquet('tables/city_69_76_final.parquet')

era_df = pd.read_csv('tables/era.csv')
year_df = pd.read_csv('tables/year.csv')

person_df = pd.read_parquet('tables/new_unified_person_df_final.parquet')
person_sentby_df = pd.read_csv('tables/person_sentby_69_76.csv')
person_sentto_df = pd.read_csv('tables/person_sentto_69_76.csv')
#person_mentioned_df = pd.read_csv('tables/person_mentioned_single_volume.csv')

religion_df = pd.read_parquet('tables/person_religion_69_76.parquet')
citizenship_df = pd.read_parquet('tables/person_citizenship_69_76.parquet')
occupation_df = pd.read_parquet('tables/person_occupation_69_76.parquet')
political_party_df = pd.read_parquet('tables/person_political_party_69_76.parquet')
role_df = pd.read_parquet('tables/person_role_69_76.parquet')
school_df = pd.read_parquet('tables/person_school_69_76.parquet')

redaction_df = pd.read_parquet('tables/redaction_69_76.parquet')
topic_desc_df = pd.read_csv('tables/topic_descp_69_76.csv')
doc_topic_df = pd.read_csv('tables/doc_topic_69_76.csv')


#graph = Graph(scheme="bolt", host="localhost", port=7687,  auth=('neo4j', 'bos'), name='neo4j')
graph = Graph(scheme="bolt", host="localhost", port=7687,  auth=('neo4j', 'bos'), name='frusphase2')

graph.delete_all()  # reset graph (only wehn first creating the databse, here for debugging purposes)


# Now neo4j does not support the numpy dtype int64, so we need to convert it to python native int
# We create a wrapper for this.
@register_attribute_postprocessor
def INT(attribute):
    # check if field is Nan
    if isinstance(attribute.value, float) and math.isnan(attribute.value):
        return Attribute(attribute.key, attribute.value)
    else:
        return Attribute(attribute.key, int(attribute.value))

@register_attribute_postprocessor
def FLOAT(attribute):
    return Attribute(attribute.key, float(attribute.value))

@register_attribute_postprocessor
def AUX(attribute):
    # check if field is Nan
    if isinstance(attribute.value, float) and math.isnan(attribute.value):
        return Attribute(attribute.key, attribute.value)
    else:
        return Attribute(attribute.key, datetime.strptime(attribute.value,'%Y-%m-%d'))


@register_subgraph_preprocessor
def ONLY_CREATE_IF_EXISTS(resource: Resource, key) -> Resource:
    val = resource[key]
    if isinstance(val, float) and math.isnan(val): # check if NaN
        return None
    elif not val: # check if None
        return None
    else:
        return resource

@register_attribute_preprocessor
def EXPORT_TEXT_TO_DB(resource: Resource) -> Resource:
    text = resource["text"]
    id = resource["id_to_text"]
    sqllite_handler.execute(f"INSERT INTO transcript VALUES(?,?);", (id,text))
    return resource


# In the schema file wrap the Person.ID attribute in the INT wrapper
#        + ID = INT(Person.ID)


iterator = IteratorIterator([PandasDataframeIterator(doc_df, "Document"), 
                             PandasDataframeIterator(era_df, "Era"), 
                             PandasDataframeIterator(person_df, "Person"),
                             PandasDataframeIterator(year_df, "Year"),
                             PandasDataframeIterator(person_sentby_df, "PersonSentBy"),
                             PandasDataframeIterator(person_sentto_df, "PersonSentTo"),
                             #PandasDataframeIterator(person_mentioned_df, "PersonMentionedIn"),
                             PandasDataframeIterator(country_df, "Country"),
                             PandasDataframeIterator(city_country_df, "CityCountry"),
                             PandasDataframeIterator(religion_df, "Religion"),
                             PandasDataframeIterator(occupation_df, "Occupation"),
                             PandasDataframeIterator(political_party_df, "PoliticalParty"),
                             PandasDataframeIterator(role_df, "Role"),
                             PandasDataframeIterator(school_df, "School"),
                             PandasDataframeIterator(citizenship_df, "Citizenship"),
                             #PandasDataframeIterator(redaction_df, "Redaction"),
                             PandasDataframeIterator(topic_desc_df, "Topic"),
                             PandasDataframeIterator(doc_topic_df, "DocTopic"),
                            ])


converter = Converter(load_file(filename), iterator, graph, num_workers=12)


converter()

# Quit sqlite handler
sqllite_handler.quit()
logging.shutdown()
print('done')