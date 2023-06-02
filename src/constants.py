# all files
# only use documents within these years
START_YEAR, END_YEAR = 1861, 1988

# define path to save extracted files
TABLES_PATH = '../tables/tables_'+str(START_YEAR)+'_'+str(END_YEAR)+'/'
PLOTS_PATH = '../plots/plots_'+str(START_YEAR)+'_'+str(END_YEAR)+'/'

# person_unify.py
PERSON_TYPO_THRESHOLD = 1
PERSON_JARO_THRESHOLD = 0.9
PERSON_LEVN_THRESHOLD = 5

# term_unify.py
TERM_TYPO_THRESHOLD = 2

# lda_topic_extraction.py
LDA_TOPIC_COUNT = 50
LDA_NAME_EXTENSION = '_lda_entremoved_min_word_len3'

# extract_entity_sentiments.py
UNWANTED_ENT_SENT = ['DATE', 'TIME', 'QUANTITY', 'ORDINAL',
                     'CARDINAL', 'MONEY', 'PERCENT']

# extract_entity_bins.py
BIN_SIZE = 4
MIN_ENTITY_COUNT = 50
BIN_NAME_EXTENSION = '_'+str(BIN_SIZE)+'yearbinned'
UNWANTED_ENT_BIN = ['DATE', 'TIME', 'QUANTITY', 'ORDINAL',
                    'CARDINAL', 'MONEY', 'PERCENT' 'PERSON']

# frus_conversion.py
AUTH = ('neo4j', 'bos')
DATABASE = 'frus'+str(START_YEAR)+'-'+str(END_YEAR)
IMPORT_PATH = '/Users/gokberk/Library/Application\ Support/Neo4j\ Desktop/Application/relate-data/dbmss/dbms-afff5963-5d02-49d0-9c4d-1d71ba108845/import'