from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import LdaModel, LdaMulticore
from gensim.utils import simple_preprocess
import os
import nltk
import gensim.downloader as api
import re
import logging

# self-defined package
from data import DataProcessed

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
# logging.root.setLevel(level=logging.INFO)
logging.root.setLevel(level=logging.ERROR)


class LDA:
    '''
    name : model name
    toload : load model from disk
    num_topics : number of topics
    workers : how many CPUs used to train LDA model
    '''
    def __init__(self,name:str = 'lda_model',toload: bool = False, num_topics=10, workers=8):
        self.name = name
        if toload:
            self.model = LdaModel.load(self.name)
        else:
            common_dictionary,common_corpus = DataProcessed().load_corpus()
            self.model = LdaMulticore(common_corpus,id2word=common_dictionary, num_topics=num_topics, workers=workers)
        
        print(self.model.print_topics(-1))



    def save(self):
        if self.model:
            self.model.save(self.name)