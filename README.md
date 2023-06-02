# decipher-frus
Deciphering the U.S. Diplomatic Documents with NLP and Graph Data Science

Codebase for Master's Thesis at ETH Zurich


## Instructions
+ Create a folder named ```volumes``` in this repo.

+ Download [FRUS files](https://github.com/HistoryAtState/frus/volumes), and put them under ```volumes```.

+ Using Python 3.8.8, run the following commands.

```
python3 -m venv frus_env
source frus_env/bin/activate
pip install -r requirements.txt
cd src/
```

+ Go to ```constants.py```, and change START_YEAR, and END_YEAR parameters depending on the experimentation range you seek. Plus, other parameters if necessary.

+ Download [Neo4j Desktop](https://neo4j.com/download/). Create a DBMS, activate it, and create database named ```frus{START_YEAR}-{END_YEAR}``` under it. Enter selected credientials AUTH parameter in ```constants.py```. 

+ Follow [this link](https://neo4j.com/docs/getting-started/appendix/tutorials/guide-import-desktop-csv/#csv-location) to reach your unique Neo4j Desktop import folder. Copy its path, and paste to IMPORT_PATH parameter in ```constants.py```.

+ Download [world_cities.csv](https://github.com/datasets/world-cities/blob/master/data/world-cities.csv), and put it under ```tables```. It is required for city-country matching (please see report).

+ Run the parsing, enrichment, and KG population files in the following order:
```
python person_unify.py
python term_unify.py
python city_country_extraction.py
python document_extraction.py
python extract_person_extras.py
python bert_topic_extraction.py --topic_count 300 --use_embeddings False --remove_entities False
python bert_topic_extraction.py --topic_count 100 --use_embeddings False --remove_entities True --name_extension _entremoved
python lda_topic_extraction.py
python redaction_extraction.py
python extract_entity_bins.py
python extract_entity_sentiments.py
python frus_conversion.py
```
Your FRUS KG is ready!

Note: ```python bert_topic_extraction.py``` requires GPU. Change ```--use_embeddings``` to True, for each option (```--remove_entities``` True or False) when ran each once.
 
+ For Redaction Analysis, follow instructions in ```src/cypher_commands.txt``` part A.

+ For Role and Person Importance Scores, follow instructions in ```src/cypher_commands.txt``` part B.

+ For Dynamic Entity Embeddings, follow instructions in ```src/cypher_commands.txt``` part C.

+ For Knowledge Graph Augmentation, run ```python link_prediction.py``` Then, follow instructions in ```src/cypher_commands.txt``` part D.

+ We provide Neo4j dump covering FRUS years from 1952 to 1988, that is ready to [download](https://polybox.ethz.ch/index.php/s/p6V43kgdNfuUI94) and analyze in Neo4j.