from graphdatascience import GraphDataScience
import pandas as pd
import os
import constants

tables_path = constants.TABLES_PATH
start_year, end_year = str(constants.START_YEAR), str(constants.END_YEAR)

# parquet to csv conversion of final tables
temp_df = pd.read_parquet(tables_path+'doc.parquet')
temp_df.to_csv(tables_path+'doc.csv')

temp_df = pd.read_parquet(tables_path+'city.parquet')
temp_df.to_csv(tables_path+'city.csv')

temp_df = pd.read_parquet(tables_path+'unified_term_df.parquet')
temp_df.to_csv(tables_path+'unified_term_df.csv')

temp_df = pd.read_parquet(tables_path+'unified_person_df_final.parquet')
temp_df.to_csv(tables_path+'unified_person_df_final.csv')

temp_df = pd.read_parquet(tables_path+'redaction.parquet')
temp_df.to_csv(tables_path+'redaction.csv')

temp_df = pd.read_parquet(tables_path+'person_school.parquet')
temp_df.to_csv(tables_path+'person_school.csv')

temp_df = pd.read_parquet(tables_path+'person_role.parquet')
temp_df.to_csv(tables_path+'person_role.csv')

temp_df = pd.read_parquet(tables_path+'person_religion.parquet')
temp_df.to_csv(tables_path+'person_religion.csv')

temp_df = pd.read_parquet(tables_path+'person_political_party.parquet')
temp_df.to_csv(tables_path+'person_political_party.csv')

temp_df = pd.read_parquet(tables_path+'person_occupation.parquet')
temp_df.to_csv(tables_path+'person_occupation.csv')

temp_df = pd.read_parquet(tables_path+'person_citizenship.parquet')
temp_df.to_csv(tables_path+'person_citizenship.csv')

temp_df = pd.read_parquet(tables_path+'entity_4yearbinned.parquet')
temp_df.to_csv(tables_path+'entity_4yearbinned.csv')

temp_df = pd.read_parquet(tables_path+'entity_sentiment.parquet')
temp_df.to_csv(tables_path+'entity_sentiment.csv')

print('parquet to csv conversion done.')

# csv to neo4j import folder transfer command.
# change the source and target directories.
source = tables_path + '*.csv'
target = constants.IMPORT_PATH

os.system("scp -r ../tables/era.csv " + target)
os.system("scp -r " + source + " " + target)

print('csv files copied to neo4j import folder.')

# extremely fast graph population with Neo4j's LOAD CSV
# connect to neo4j database
gds = GraphDataScience("bolt://localhost:7687", auth=constants.AUTH, 
                       database=constants.DATABASE)

print('connected to KG. population starting...')

# create indices
gds.run_cypher(
    """
    CREATE INDEX doc_index FOR (d:Document) ON (d.docID)
    """)
gds.run_cypher(
    """
    CREATE INDEX person_index FOR (p:Person) ON (p.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX city_index FOR (c:City) ON (c.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX era_index FOR (e:PresidentialEra) ON (e.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX country_index FOR (c:Country) ON (c.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX occupation_index FOR (o:Occupation) ON (o.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX school_index FOR (s:School) ON (s.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX party_index FOR (p:PoliticalParty) ON (p.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX role_index FOR (r:Role) ON (r.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX religion_index FOR (r:Religion) ON (r.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX bert_topic_index FOR (t:TopicBertWithEntities) ON (t.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX bert_topic_no_entity_index FOR (t:TopicBertNoEntities) ON (t.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX lda_topic_no_entity_index FOR (t:TopicLDANoEntities) ON (t.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX dynamic_entity_4_index FOR (dne:DynamicEntity4YearBinned) ON (dne.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX term_index FOR (t:Term) ON (t.name)
    """)
gds.run_cypher(
    """
    CREATE INDEX redaction_index FOR (r:Redaction) ON (r.redaction_id)
    """)
gds.run_cypher(
    """
    CREATE INDEX named_entity_index FOR (ne:NamedEntity) ON (ne.name)
    """)

# load csv files
gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///era.csv' AS row
    CALL {
        with row
        merge (e:PresidentialEra {name:row.president})
        on create set 
            e.startDate = date(row.startDate),
            e.endDate = date(row.endDate),
            e.startYear = toInteger(row.startYear),
            e.endYear = toInteger(row.endYear)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///country.csv' AS row
    CALL {
        with row
        merge (c:Country {name:row.countryLabel})
        on create set 
            c.tag = row.countryTag
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///city.csv' AS row
    CALL {
        with row
        merge (ci:City {name:row.placeName})
        with row, ci
        match (c:Country {name:row.country})
        merge (ci)-[:LOCATED_IN]->(c)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///doc.csv' AS row
    CALL {
        with row
        merge (d:Document {docID:row.id_to_text})
        on create set 
            d.subtype = row.subtype,
            d.volume = row.volume,
            d.date = date(row.date),
            d.year = toInteger(row.year),
            d.text_length = toInteger(row.txt_len),
            d.subjectivity = toFloat(row.subj),
            d.polarity = toFloat(row.pol),
            d.type_token_ratio = toFloat(row.ttr),
            d.corrected_type_token_ratio = toFloat(row.cttr)
        with row, d
        match (e:PresidentialEra {name:row.era})
        merge (d)-[:DURING]->(e)
        with row, d
        match (c:City {name:row.city})
        merge (d)-[:FROM]->(c)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///unified_person_df_final.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        on create set 
            p.name_list = row.name_list,
            p.id_list = row.id_list,
            p.description_list = row.description_list,
            p.candidate_wiki_entries = row.wiki_col,
            p.selected_wiki_entity = row.selected_wiki_entity,
            p.gender = row.gender
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_sentby.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.sent})
        match (p:Person {name:row.person_name})
        merge (d)-[:SENT_BY]->(p)
        } IN TRANSACTIONS OF 100000 ROWS;
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_sentto.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.received})
        match (p:Person {name:row.person_name})
        merge (d)-[:SENT_TO]->(p)
        } IN TRANSACTIONS OF 100000 ROWS;
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_mentioned.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.mentioned_in})
        match (p:Person {name:row.person_name})
        merge (d)-[:MENTIONED]->(p)
        } IN TRANSACTIONS OF 100000 ROWS;
        """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_citizenship.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (c:Country {name:row.info_name})
        merge (p)-[r:CITIZEN_OF]->(c)
        on create set
        c.tag = row.info_tag,
        r.started = row.start_year,
        r.ended = row.end_year
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_occupation.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (o:Occupation {name:row.info_name})
        merge (p)-[r:WORKED_AS]->(o)
        on create set
        o.tag = row.info_tag
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_school.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (s:School {name:row.info_name})
        merge (p)-[r1:EDUCATED_AT]->(s)
        on create set
        s.tag = row.info_tag
        with row, s
        where row.country is not null
        merge (c:Country {name:row.country})
        merge (s)-[r2:IN]->(c)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_political_party.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (po:PoliticalParty {name:row.info_name})
        merge (p)-[r1:MEMBER_OF]->(po)
        on create set
        po.tag = row.info_tag,
        r1.started = row.start_year,
        r1.ended = row.end_year
        with row, po
        where row.country is not null
        merge (c:Country {name:row.country})
        merge (po)-[r2:IN]->(c)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_role.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (r:Role {name:row.info_name})
        merge (p)-[rel:POSITION_HELD]->(r)
        on create set
        r.tag = row.info_tag,
        rel.started = row.start_year,
        rel.ended = row.end_year
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///person_religion.csv' AS row
    CALL {
        with row
        merge (p:Person {name:row.name_set})
        merge (r:Religion {name:row.info_name})
        merge (p)-[:BELIEVED]->(r)
        on create set
        r.tag = row.info_tag
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///redaction.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (r:Redaction {redaction_id:toInteger(row.redaction_id)})
        on create set 
            r.raw_text = row.raw_text,
            r.detected_chunks = row.detected_chunks,
            r.type = row.type_col,
            r.amount = toFloat(row.amount_col)
        merge (d)-[:REDACTED]->(r)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///topic_descp.csv' AS row
    CALL {
        with row
        merge (t:TopicBertWithEntities {name:row.Name})
        on create set
        t.description = row.Top_n_words
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///doc_topic.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (t:TopicBertWithEntities {name:row.assigned_topic})
        merge (d)-[:ABOUT]->(t)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///topic_descp_entremoved.csv' AS row
    CALL {
        with row
        merge (t:TopicBertNoEntities {name:row.Name})
        on create set
        t.description = row.Top_n_words
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///doc_topic_entremoved.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (t:TopicBertNoEntities {name:row.assigned_topic})
        merge (d)-[:ABOUT]->(t)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///entity_4yearbinned.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (dne:DynamicEntity4YearBinned {name:row.dynamic_named_entity})
        merge (d)-[:MENTIONED]->(dne)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///unified_term_df.csv' AS row
    CALL {
        with row
        merge (t:Term {name:row.description_set})
        on create set 
            t.name_list = row.description_list,
            t.id_list = row.id_list
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///term_mentioned.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.mentioned_in})
        match (t:Term {name:row.description_set})
        merge (d)-[:MENTIONED]->(t)
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)


# NOT AVAILABLE CURRENTLY, remove comments if available.
'''
gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///topic_descp_lda_entremoved_min_word_len3.csv' AS row
    CALL {
        with row
        merge (t:TopicLDANoEntities {name:row.Name})
        on create set
        t.description = row.Top_n_words
        } IN TRANSACTIONS OF 100000 ROWS; 
    """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///doc_topic_lda_entremoved_min_word_len3.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (t:TopicLDANoEntities {name:row.assigned_topic})
        merge (d)-[:ABOUT]->(t)
        } IN TRANSACTIONS OF 100000 ROWS; 
        """)

gds.run_cypher(
    """
    LOAD CSV WITH HEADERS FROM 'file:///entity_sentiment.csv' AS row
    CALL {
        with row
        match (d:Document {docID:row.id_to_text})
        merge (ne:NamedEntity {name:row.named_entity})
        merge (d)-[r:HAS_SENTIMENT]->(ne)
        on create set
        r.polarity = row.pol,
        r.subjectivity = row.sub
        } IN TRANSACTIONS OF 100000 ROWS; 
        """)
'''

print('KG population done.')