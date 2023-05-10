import sys
import ssl
import math
import logging
import sqlite3
import sqllite_handler
import pandas as pd
from datetime import datetime
from py2neo import Graph
from rel2graph.relational_modules.pandas import PandasDataframeIterator
from rel2graph import IteratorIterator
from rel2graph import Converter
from rel2graph.utils import load_file
from rel2graph import register_attribute_postprocessor, Attribute, \
 register_attribute_preprocessor, Resource, register_subgraph_preprocessor

# Hack to make the module importable
sys.path.append(r'./../')
ssl._create_default_https_context = ssl._create_unverified_context

# configure logging
# logging.basicConfig(filename='tables/rel2graphlogs.log',level=logging.WARNING)
logger = logging.getLogger("rel2graph")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    "%(asctime)s [%(threadName)s]::[%(levelname)s]::%(filename)s: %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# necessary variable definitions
filename = "frus_schema.yaml"
tables_path = 'tables/tables_52_88/'
rdb_name = tables_path+'texts.db'


if __name__ == "__main__":

    # Setup sqlite database for transcript texts
    # Create table if not exists
    print("Setting up sqlite database.")
    conn = sqlite3.connect(rdb_name)

    conn.execute('''CREATE TABLE IF NOT EXISTS transcript
            (ID PRIMARY KEY NOT NULL,
            TEXT,
            KEYWORDS);''')
    conn.close()

    sqllite_handler.init(rdb_name)

    # load all files
    # document files
    doc_df = pd.read_csv(tables_path+'doc.csv')
    # change year from type 'float' to 'str(int)' suitable for rel2graph
    doc_df['year'] = doc_df['year'].apply(lambda x: x if math.isnan(x) 
                                          else str(int(x)))

    era_df = pd.read_csv('tables/era.csv')

    # city country files
    country_df = pd.read_csv(tables_path+'country.csv')
    city_country_df = pd.read_parquet(tables_path+'city.parquet')
    # country_mentioned_df = pd.read_csv(tables_path+'country_mentioned.csv')

    # person files
    person_df = pd.read_parquet(tables_path+'unified_person_df_final.parquet')
    person_sentby_df = pd.read_csv(tables_path+'person_sentby.csv')
    person_sentto_df = pd.read_csv(tables_path+'person_sentto.csv')
    person_mentioned_df = pd.read_csv(tables_path+'person_mentioned.csv')

    # person extras files
    religion_df = pd.read_parquet(tables_path+'person_religion.parquet')
    citizenship_df = pd.read_parquet(tables_path+'person_citizenship.parquet')
    occupation_df = pd.read_parquet(tables_path+'person_occupation.parquet')
    political_party_df = pd.read_parquet(
        tables_path+'person_political_party.parquet')
    role_df = pd.read_parquet(tables_path+'person_role.parquet')
    school_df = pd.read_parquet(tables_path+'person_school.parquet')

    # term files
    term_df = pd.read_parquet(tables_path+'unified_term_df.parquet')
    term_mentioned_df = pd.read_csv(tables_path+'instution_mentioned.csv')

    # redaction file
    redaction_df = pd.read_parquet(tables_path+'redaction.parquet')

    # topic files
    topic_desc_BertWithEntities_df = pd.read_csv(tables_path+'topic_descp.csv')
    doc_topic_BertWithEntities_df = pd.read_csv(tables_path+'doc_topic.csv')
    topic_desc_BertNoEntities_df = pd.read_csv(
        tables_path+'topic_descp_entremoved.csv')
    doc_topic_BertNoEntities_df = pd.read_csv(
        tables_path+'doc_topic_entremoved.csv')
    topic_desc_LDANoEntities_df = pd.read_parquet(
        tables_path+'topic_descp_lda_entremoved_len3.parquet')
    doc_topic_LDANoEntities_df = pd.read_parquet(
        tables_path+'doc_topic_lda_entremoved_len3.parquet')

    # named entity files and their sentiments
    # entity_sent_df = pd.read_parquet(tables_path+'entity_sentiment.parquet')

    # dynamic named entity files for dynamic named entity embeddings
    # change path manually each time!  <x>yearbinned, x user input.
    dynamic_entity4year_doc_df = pd.read_parquet(
        tables_path+'ne2doc_4yearbinned.parquet') 
    # dynamic_entity5year_doc_df = pd.read_parquet(
    # tables_path+'ne2doc_5yearbinned.parquet')

    # person-person similarity files for link prediction
    similar_descp_persons = pd.read_parquet(
        tables_path+'similar_descp_persons.parquet')

    # select graph to write
    graph = Graph(scheme="bolt", host="localhost", port=7687,
                  auth=('neo4j', 'bos'), name='frus5288')

    # uncomment below if writing from scratch. leave on comment if updating
    # graph.delete_all()

    # define pre and post processors
    # Now neo4j does not support the numpy dtype int64, so we need to convert 
    # it to python native int
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
            return Attribute(attribute.key,
                             datetime.strptime(attribute.value, '%Y-%m-%d'))

    @register_subgraph_preprocessor
    def ONLY_CREATE_IF_EXISTS(resource: Resource, key) -> Resource:
        val = resource[key]
        if isinstance(val, float) and math.isnan(val):  # check if NaN
            return None
        elif not val:  # check if None
            return None
        else:
            return resource

    @register_attribute_preprocessor
    def EXPORT_TEXT_TO_DB(resource: Resource) -> Resource:
        text = resource["text"]
        id = resource["id_to_text"]
        sqllite_handler.execute("INSERT INTO transcript \
                                 VALUES(?,?);", (id, text))
        return resource

    # alternative method to above where stores keywords in db as well
    @register_attribute_preprocessor
    def EXPORT_TEXT_AND_KEYWORDS_TO_DB(resource: Resource) -> Resource:
        text = resource["text"]
        id = resource["id_to_text"]
        keywords = resource["keywords"]
        sqllite_handler.execute("INSERT INTO transcript \
                                VALUES(?,?,?);", (id, text, keywords))
        return resource

    iterator = IteratorIterator(
        [PandasDataframeIterator(doc_df, "Document"), 
         # PandasDataframeIterator(era_df, "Era"), 
         # PandasDataframeIterator(person_df, "Person"),
         # PandasDataframeIterator(person_sentby_df, "PersonSentBy"),
         # PandasDataframeIterator(person_sentto_df, "PersonSentTo"),
         # PandasDataframeIterator(person_mentioned_df, "PersonMentionedIn"),
         # PandasDataframeIterator(country_df, "Country"),
         # PandasDataframeIterator(city_country_df, "CityCountry"),
         # PandasDataframeIterator(country_mentioned_df, "CountryMentionedIn"),
         # PandasDataframeIterator(religion_df, "Religion"),
         # PandasDataframeIterator(occupation_df, "Occupation"),
         # PandasDataframeIterator(political_party_df, "PoliticalParty"),
         # PandasDataframeIterator(role_df, "Role"),
         # PandasDataframeIterator(school_df, "School"),
         # PandasDataframeIterator(citizenship_df, "Citizenship"),
         # PandasDataframeIterator(redaction_df, "Redaction"),
         # PandasDataframeIterator(topic_desc_BertWithEntities_df, "TopicBertWithEntities"),
         # PandasDataframeIterator(doc_topic_BertWithEntities_df, "DocTopicBertWithEntities"),
         # PandasDataframeIterator(topic_desc_BertNoEntities_df, "TopicBertNoEntities"),
         # PandasDataframeIterator(doc_topic_BertNoEntities_df, "DocTopicBertNoEntities"),
         # PandasDataframeIterator(topic_desc_LDANoEntities_df, "TopicLDANoEntities"),
         # PandasDataframeIterator(doc_topic_LDANoEntities_df, "DocTopicLDANoEntities"),
         # PandasDataframeIterator(entity_sent_df, "EntitySentiment"),
         # PandasDataframeIterator(dynamic_entity4year_doc_df, "DocDynamicEnt4YearBinned"),
         # PandasDataframeIterator(dynamic_entity5year_doc_df, "DocDynamicEnt5YearBinned")
         # PandasDataframeIterator(similar_descp_persons, "PersonDescpSimilarity"),
        ])

    converter = Converter(load_file(filename), iterator, graph, num_workers=12)

    converter()

    # Quit sqlite handler
    sqllite_handler.quit()
    print('finished.')
    