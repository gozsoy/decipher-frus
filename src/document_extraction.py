import glob
# import json
import pandas as pd
from tqdm import tqdm
import datetime as dtm
from datetime import datetime
import xml.etree.ElementTree as ET
from lexicalrichness import LexicalRichness
from textblob import TextBlob
import ray

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


# helper function for parsing a single document from a volume into its fields
@ray.remote
def extract_document(doc, volume):

    # assign a unique id to document
    id_to_text = volume + '_' + \
                doc.attrib['{http://www.w3.org/XML/1998/namespace}id']

    # extract document subtype
    try:
        subtype = doc.attrib['subtype']
    except:
        subtype = "historical-document"

    # extract document date, year, president era
    date = None
    year = None
    era = None
    # if subtype != 'editorial-note':
    fmt = doc.attrib[
        '{http://history.state.gov/frus/ns/1.0}doc-dateTime-max']
    date = datetime.strptime(fmt.split('T')[0], '%Y-%m-%d')
    year = datetime.strptime(fmt.split('T')[0], '%Y-%m-%d').year
    try:
        era = era_df[(era_df['startDate'] <= date)
                     & (era_df['endDate'] > date)].president.values[0]
    except:
        volume_year = int(volume[4:8])
        year = volume_year
        date = dtm.datetime(volume_year, 1, 1)
        era = era_df[(era_df['startYear'] <= volume_year)
                     & (era_df['endYear'] > volume_year)].president.values[0]

    # extract document pyhsical source
    source_tag = doc.find('.//dflt:note[@type="source"]', ns)
    if source_tag is not None:
        source = " ".join(ET.tostring(source_tag, encoding='unicode',
                                      method='text').split())
    else:
        source = None

    # extract document title, removes if <note> exists!
    head_tag = doc.find('./dflt:head', ns)
    if head_tag:
        child_note_tags = head_tag.findall('./dflt:note', ns)

        for note_tag in child_note_tags:
            head_tag.remove(note_tag)

        title = " ".join(ET.tostring(head_tag,
                                     encoding='unicode',
                                     method='text').split())
    else:
        title = 'No title.'

    # extract document source city (place actually)
    place_tag = doc.find('.//dflt:placeName', ns)
    if place_tag is not None:
        txt = "".join(place_tag.itertext())
        txt = " ".join(txt.split())
        city = txt
        # txt = " ".join(txt.split(',')[0].split())
        # city = city_lookup_dict[txt]
        # city = txt
    else:
        city = None

    # extract persons sent the document
    person_sentby = []
    person_sentby_dict_list = []

    # 1
    for pers_tag in doc.findall('.//dflt:persName[@type="from"]', ns):
        if pers_tag is not None:
            # if tag has corresp attr that is person identifier, proceed.
            if 'corresp' in pers_tag.attrib:
                if pers_tag.attrib['corresp'][0] == '#':  # naming convention 1
                    person_id = pers_tag.attrib['corresp'][1:]
                else:  # naming convention 2
                    person_id = pers_tag.attrib['corresp']
                # look up corresponding person from unified persons
                person_id = volume + '_' + person_id
                person_name = person_id_lookup_dict.get(person_id, None)
                if person_name:
                    person_sentby.append(person_name)
                    person_sentby_dict_list.append({'person_name': person_name,
                                                    'sent': id_to_text}) 
            # if tag is not person annotated i.e. just a plain string
            else:
                txt = (" ".join(pers_tag.itertext()))
                txt = " ".join(txt.split())
                person_sentby.append(txt)

    # 2 also consider persons signed the document as persons sent it
    signed_person_tag = doc.find('.//dflt:signed//dflt:persName', ns)
    if signed_person_tag is not None:
        if 'corresp' in signed_person_tag.attrib:
            person_id = signed_person_tag.attrib['corresp'][1:]
            if signed_person_tag.attrib['corresp'][0] == '#':
                person_id = signed_person_tag.attrib['corresp'][1:]
            else:
                person_id = signed_person_tag.attrib['corresp']
            person_id = volume + '_' + person_id
            person_name = person_id_lookup_dict.get(person_id, None)
            if person_name:
                person_sentby.append(person_name)
                person_sentby_dict_list.append({'person_name': person_name, 
                                                'sent': id_to_text})
        else:
            txt = (" ".join(signed_person_tag.itertext()))
            txt = " ".join(txt.split())
            person_sentby.append(txt)

    # extract persons received the document
    person_sentto = []
    person_sentto_dict_list = []

    # same code structure with person_sentby (above)
    for pers_tag in doc.findall('.//dflt:persName[@type="to"]', ns):
        if pers_tag is not None:
            if 'corresp' in pers_tag.attrib:
                if pers_tag.attrib['corresp'][0] == '#':
                    person_id = pers_tag.attrib['corresp'][1:]
                else:
                    person_id = pers_tag.attrib['corresp']
                person_id = volume + '_' + person_id
                person_name = person_id_lookup_dict.get(person_id, None)
                if person_name:
                    person_sentto.append(person_name)
                    person_sentto_dict_list.append({'person_name': person_name,
                                                    'received': id_to_text})
            else:
                txt = (" ".join(pers_tag.itertext()))
                txt = " ".join(txt.split())
                person_sentto.append(txt)

    # extract institutions sent the document
    inst_sentby = []

    # these only appear as plain string, no annotation
    for gloss_tag in doc.findall('.//dflt:gloss[@type="from"]', ns):

        txt = (" ".join(gloss_tag.itertext()))
        txt = " ".join(txt.split())
        inst_sentby.append(txt)

    # extract institutions received the document
    inst_sentto = []

    # these only appear as plain string, no annotation
    for gloss_tag in doc.findall('.//dflt:gloss[@type="to"]', ns):

        txt = (" ".join(gloss_tag.itertext()))
        txt = " ".join(txt.split())
        inst_sentto.append(txt)

    # extract persons mentioned in document, not counts <note>!
    person_mentioned = set()
    person_mentioned_dict_list = []

    # remove <note> because they are simply footnotes or sources
    notes_parent_tags = doc.findall('.//dflt:note/..', ns)

    for parent_tag in notes_parent_tags:

        for note_tag in parent_tag.findall('./dflt:note', ns):
            parent_tag.remove(note_tag)

    # same code structure as in person_sentby, and person_sentto
    pers_tags = doc.findall('.//dflt:persName[@corresp]', ns)
    for temp_tag in pers_tags:
        if temp_tag.attrib['corresp'][0] == '#':
            person_id = temp_tag.attrib['corresp'][1:]
        else:
            person_id = temp_tag.attrib['corresp']
        person_id = volume + '_' + person_id
        person_name = person_id_lookup_dict.get(person_id, None)
        if person_name:
            person_mentioned.add(person_name)
            person_mentioned_dict_list.append({'person_name': person_name,
                                               'mentioned_in': id_to_text})

    # extract terms mentioned in document
    instution_mentioned = set()
    institution_mentioned_dict_list = []

    # same code structure as in person_sentby, and person_sentto
    inst_tags = doc.findall('.//dflt:gloss[@target]', ns)
    for temp_tag in inst_tags:
        if temp_tag.attrib['target'][0] == '#':
            term_id = temp_tag.attrib['target'][1:]
        else:
            term_id = temp_tag.attrib['target']
        term_id = volume + '_' + term_id
        term_description_set = institution_id_lookup_dict.get(term_id, None)
        if term_description_set:
            instution_mentioned.add(term_description_set)
            institution_mentioned_dict_list.append({
                'description_set': term_description_set,
                'mentioned_in': id_to_text})

    # extract document's free text
    free_text = ""

    tag_list = doc.findall('./*', ns)
    
    # find free text's start and end elements
    lidx, ridx = 0, 0

    for idx, tag in enumerate(tag_list):
        if tag.tag not in not_text_tags:
            lidx = idx
            break
    
    for idx, tag in enumerate(tag_list[::-1]):
        if tag.tag in text_tags:
            ridx = len(tag_list)-1-idx
            break
    
    # remove all <note> in free text
    notes_parent_tags = doc.findall('.//dflt:note/..', ns)

    for parent_tag in notes_parent_tags:
        for note_tag in parent_tag.findall('./dflt:note', ns):
            parent_tag.remove(note_tag)

    # join free text pieces
    for f_tag in tag_list[lidx:ridx+1]:
        free_text += " ".join("".join(f_tag.itertext()).split()) + " "
    
    # if after all, free text is still "" represent document with "-" 
    # to deal with nan values later.
    if free_text == "":
        free_text = "-"
    
    # compute string measures (lexical richness, polarity, token count)
    blob = TextBlob(free_text)
    lex = LexicalRichness(free_text)
    txt_len = lex.words
    subj = round(blob.sentiment[1], 2)
    pol = round(blob.sentiment[0], 2)
    if txt_len != 0:
        ttr = round(lex.ttr, 2)
        cttr = round(lex.cttr, 2)
    else:
        ttr = 0
        cttr = 0

    # merge all extracted info into one
    doc_dict = {'id_to_text': id_to_text, 'volume': volume, 'subtype': subtype,
                'date': date, 'year': year, 'title': title,
                'source': source, 'person_sentby': person_sentby,
                'person_sentto': person_sentto,
                'city': city, 'era': era, 'inst_sentby': inst_sentby,
                'inst_sentto': inst_sentto,
                'person_mentioned': person_mentioned,
                'inst_mentioned': instution_mentioned, 'text': free_text,
                'txt_len': txt_len, 'subj': subj, 'pol': pol, 'ttr': ttr,
                'cttr': cttr,
                }
    
    # return extracted information in a tuple all elements with some duty
    return (person_sentby_dict_list, person_sentto_dict_list, 
            person_mentioned_dict_list, institution_mentioned_dict_list,
            doc_dict)


if __name__ == "__main__":

    # initialize parallel operation
    ray.init(num_cpus=13)

    # city lookup table for unification
    # with open(tables_path+'city_lookup_dict.json', 'r') as f:
    #    city_lookup_dict = json.load(f)

    # person id to unified name lookup table
    new_unified_person_df = pd.read_parquet(
        tables_path+'unified_person_df_final.parquet')

    person_id_lookup_dict = {}  # 'id':'corrected'
    for _, row in new_unified_person_df.iterrows():

        for id in row['id_list']:
            if id not in person_id_lookup_dict:
                person_id_lookup_dict[id] = row['name_set']

    # term id to unified name lookup table
    new_unified_institution_df = pd.read_parquet(
        tables_path+'unified_term_df.parquet')

    institution_id_lookup_dict = {}  # 'id':'corrected'
    for _, row in new_unified_institution_df.iterrows():

        for id in row['id_list']:
            if id not in institution_id_lookup_dict:
                institution_id_lookup_dict[id] = row['description_set']

    # defining useful tag lists for free text's extraction
    not_text_tags = ['{http://www.tei-c.org/ns/1.0}head',
                     '{http://www.tei-c.org/ns/1.0}opener',
                     '{http://www.tei-c.org/ns/1.0}dateline',
                     '{http://www.tei-c.org/ns/1.0}note',
                     '{http://www.tei-c.org/ns/1.0}table']
    text_tags = ['{http://www.tei-c.org/ns/1.0}p',
                 '{http://www.tei-c.org/ns/1.0}list']

    # compute presidential era start and end dates
    era_df = pd.read_csv('../tables/era.csv')
    era_df['startDate'] = era_df['startDate'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d'))
    era_df['endDate'] = era_df['endDate'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d'))

    # variables to merge information from all volumes
    global_person_sentby_list = []
    global_person_sentto_list = []
    global_person_mentioned_list = []
    global_institution_mentioned_list = []
    global_doc_list = []

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
            # parallely process each single document within volume
            futures = [extract_document.remote(doc, volume) for doc in docs]
            result_tuple_list = ray.get(futures)

            # merge volume results with global variables
            global_person_sentby_list += sum(list(map(
                lambda x: x[0], result_tuple_list)), [])
            global_person_sentto_list += sum(list(map(
                lambda x: x[1], result_tuple_list)), [])
            global_person_mentioned_list += sum(list(map(
                lambda x: x[2], result_tuple_list)), [])
            global_institution_mentioned_list += sum(list(map(
                lambda x: x[3], result_tuple_list)), [])
            global_doc_list += list(map(lambda x: x[4], result_tuple_list))

    # close parallel processes
    ray.shutdown()

    # save results in tables path
    doc_df = pd.DataFrame(global_doc_list)
    person_sentby_df = pd.DataFrame(global_person_sentby_list)
    person_sentto_df = pd.DataFrame(global_person_sentto_list)
    person_mentioned_df = pd.DataFrame(global_person_mentioned_list)
    instution_mentioned_df = pd.DataFrame(global_institution_mentioned_list)

    doc_df.to_parquet(tables_path+'doc.parquet')
    person_sentby_df.to_csv(tables_path+'person_sentby.csv')
    person_sentto_df.to_csv(tables_path+'person_sentto.csv')

    # remove duplicate rows
    person_mentioned_df = person_mentioned_df[['person_name', 'mentioned_in']]\
        .drop_duplicates().reset_index(drop=True)
    person_mentioned_df.to_csv(tables_path+'person_mentioned.csv')

    instution_mentioned_df = instution_mentioned_df[['description_set', 
                                                     'mentioned_in']]\
        .drop_duplicates().reset_index(drop=True)
    instution_mentioned_df.to_csv(tables_path+'term_mentioned.csv')
    
    print('finished.')