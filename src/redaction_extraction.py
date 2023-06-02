import re
import glob
import spacy
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import ray
import constants
import os

# define namespaces in FRUS schema
ns = {'xml': 'http://www.w3.org/XML/1998/namespace',
      'dflt': 'http://www.tei-c.org/ns/1.0',
      'frus': 'http://history.state.gov/frus/ns/1.0',
      'xi': 'http://www.w3.org/2001/XInclude'
      }

tables_path = constants.TABLES_PATH
start_year, end_year = constants.START_YEAR, constants.END_YEAR

nlp = spacy.load('en_core_web_sm')

if not os.path.exists(tables_path):
    os.makedirs(tables_path)


# helper function 1 step 1
# helper function for parsing redactions in a document
@ray.remote
def extract_redaction(doc, volume):

    doc_redaction_list = []

    # id
    id_to_text = volume + '_' +\
        doc.attrib['{http://www.w3.org/XML/1998/namespace}id']

    # redaction text and amount
    for el in doc.findall('.//dflt:hi[@rend="italic"]', ns):
        temp_txt = "".join(el.itertext())
        temp_txt = " ".join(temp_txt.split())  # remove \n
        if re.search('not declassified', temp_txt):  # if redaction identified

            chunks = []
            doc = nlp(temp_txt)
            for chunk in doc.noun_chunks:
                chunks.append("".join(chunk.text))
                
            doc_redaction_list.append({'id_to_text': id_to_text,
                                       'raw_text': temp_txt,
                                       'detected_chunks': chunks})
    
    return doc_redaction_list


# helper function 1 step 2
# converts a str to float if possible
def convert_str2float(x):
    try:
        return float(eval(x))
    except:
        return None


if __name__ == "__main__":

    #####
    # PART 1: EXTRACT REDACTIONS
    #####

    # initialize parallel operation
    ray.init(num_cpus=13)

    # variables to merge information from all volumes
    global_redaction_list = []

    # main loop over all volumes
    for file in tqdm(glob.glob('../volumes/frus*')):
        file_start_year = int(file[15:19])
        
        # if volume date is within specified dates
        if file_start_year >= start_year and file_start_year <= end_year:

            volume = file[11:-4]

            tree = ET.parse(file)
            root = tree.getroot()

            # find all documents in volume
            docs = root.findall(
                './dflt:text/dflt:body//dflt:div[@type="document"]', ns)

            futures = [extract_redaction.remote(doc, volume) for doc in docs]
            result_tuple_list = ray.get(futures)

            global_redaction_list += sum(result_tuple_list, [])
    
    # close parallel processes
    ray.shutdown()

    # convert results to pd dataframe
    redaction_df = pd.DataFrame(global_redaction_list)

    #####
    # PART 2: PROCESS EXTRACTED REDACTIONS
    #####

    # count redaction type's frequencies for unification
    type_dict = {}

    for idx, temp_text in enumerate(tqdm(redaction_df['raw_text'])):

        # this symbol is problematic, remove it
        temp_text = temp_text.replace('½', '')
        
        doc = nlp(temp_text)

        for token in doc:
            if token.pos_ == 'NOUN':
                cnt = type_dict.get(token.lemma_, 0)+1
                type_dict[token.lemma_] = cnt

    # process each redaction's raw text, and extract its type and amount
    type_col = []
    amount_col = []

    for idx, temp_text in enumerate(tqdm(redaction_df['raw_text'])):

        # find paranthesis in raw redaction text, if exist
        result = re.findall('\((.*?)\)', temp_text)
        # only use first paranthesis further
        if len(result) == 1:
            temp_text = result[0]
        # resolve multi-paranthesis cases by hand.
        # prints these for debugging purposes
        elif len(result) > 1:
            print(f'Untidy reduction format.'
                  f'Multi-paranthesis use in row {idx}: {temp_text}')

        # this symbol is problematic, remove it
        temp_text = temp_text.replace('½', '') 
        doc = nlp(temp_text)

        # select most relevant noun chunk
        chunk = None
        if len(list(doc.noun_chunks)) == 1:
            chunk = list(doc.noun_chunks)[0]
        elif len(list(doc.noun_chunks)) > 1:
            max_count = -1 

            for temp_chunk in doc.noun_chunks:
                for token in temp_chunk:
                    if token.pos_ == 'NOUN':
                        temp_count = type_dict.get(token.lemma_, 0)

                        if temp_count > max_count:
                            max_count = temp_count
                            chunk = temp_chunk

        # separate type and amount for selected noun chunk, if possible
        if chunk is None:
            type_col.append(None)
            amount_col.append(None)
        else:       
            type_ = ''
            amount = ''
            for token in chunk:
                if token.like_num:
                    amount += token.text
                elif token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                    type_ += token.lemma_   
            
            type_col.append(type_)
            amount_col.append(amount)

    redaction_df['type_col'] = type_col
    redaction_df['amount_col'] = amount_col

    redaction_df['amount_col'] = redaction_df['amount_col'].apply(
        lambda x: convert_str2float(x))

    redaction_df.reset_index(drop=False, inplace=True)
    redaction_df.rename(columns={'index': 'redaction_id'}, inplace=True)
    redaction_df.to_parquet(tables_path+'redaction.parquet')

    print('finished.')


