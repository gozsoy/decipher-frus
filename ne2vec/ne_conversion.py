if __name__ == "__main__":

    import sys
    import ssl
    import logging
    import numpy as np
    import pandas as pd

    from py2neo import Graph, NodeMatcher
    from rel2graph.relational_modules.pandas import PandasDataframeIterator
    from rel2graph import IteratorIterator
    from rel2graph import Converter
    from rel2graph.utils import load_file
    
    sys.path.append(r'./../')
    ssl._create_default_https_context = ssl._create_unverified_context

    logger = logging.getLogger("rel2graph")
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s]::[%(levelname)s]::%(filename)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)


    filename = "ne_schema.yaml"

    ne2doc_df = pd.read_parquet('ne2doc_df.parquet')

    graph = Graph(scheme="bolt", host="localhost", port=7687,  auth=('neo4j', 'bos'), name='entity2vec')

    graph.delete_all()

    iterator = IteratorIterator([PandasDataframeIterator(ne2doc_df, "Doc2Ent")])

    converter = Converter(load_file(filename), iterator, graph, num_workers=12)

    converter()

    print('done')