# decipher-frus
Deciphering the U.S. Cables with NLP and Graph Data Science

Codebase for Master's Thesis at ETH Zurich


## Instructions
+ Download [Neo4j Desktop](https://neo4j.com/download/). Create a DBMS, activate it, and create database under it. Enter selected credientials to line 114 in frus_conversion.py. 

+ Create folders named 'volumes', and 'plots' in this repo.

+ Download [FRUS files](https://github.com/HistoryAtState/frus/volumes), and put them under 'volumes'.

+ Using Python 3.8.8, run the following commands.

```
python3 -m venv frus_env
source frus_env/bin/activate
pip install -r requirements.txt
cd src/
```

+ Download [world_cities.csv](https://github.com/datasets/world-cities/blob/master/data/world-cities.csv), and put it under 'tables'. It is required for city-country matching (please see report).

+ Run the parsing, enrichment, and graph population files in the following order.

```
python person_unify.py
python term_unify.py
python city_country_extraction.py
python document_extraction.py
python extract_person_extras.py
python bert_topic_extraction.py
python lda_topic_extraction.py
python redaction_extraction.py
python extract_entity_bins.py
python extract_entity_sentiments.py
python frus_conversion.py
```
Your FRUS KG is ready!

Note 1: Do not forget to comment line 118 in frus_conversion.py when ran above once. This will ensure any change afterwards will update the existing graph.

Note 2: Do not forget to create an experiment specific subfolder under both 'tables' and 'plots' !
```
tables/tables_1952_1988
plots/plots_1952_1988
```

Note 3: Do not forget to change the file specific variables before running each file! (Will be unified in following iterations).
```
start_year, end_year = 1952, 1988
tables_path = '../tables/tables_1952_1988/'
plots_path = '../plots/plots_1952_1988/'
```

+ For Role and Person Importance Scores, follow instructions in 'src/cypher_commands.txt' part B.

+ For Dynamic Entity Embeddings, follow instructions in 'src/cypher_commands.txt' part C.

+ For Knowledge Graph Augmentation, run
```
python link_prediction.py
```
Then, follow instructions in 'src/cypher_commands.txt' D.

+ We provide Neo4j dump covering FRUS years from 1952 to 1988, that is ready to [download](https://polybox.ethz.ch/index.php/s/p6V43kgdNfuUI94) and analyze in Neo4j.