from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath,get_tmpfile
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
import re
import logging

class DataProcessed:
    def __init__(self,num_samples=1000,load_cache=True,dictName='mydict1.dict',corpusName='corpus.mm'):
        self.dictName = datapath(dictName)
        self.corpusName = datapath(corpusName)
        self.load_cache = load_cache
        self.num_samples = num_samples
        
    
    def load_corpus(self):
        # build corpus and return dictionary and corpus
        def build_corpus(num_samples=self.num_samples):
            # get stop words
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            stop_words = set(stopwords.words('english'))

            # load wiki corpus
            dataset = api.load("text8")
            data = [d for d in dataset]

            # Step 2: Prepare Data (Remove stopwords and lemmatize)
            data_processed = []

            # get lemmatizer from nltk
            lemmatizer = WordNetLemmatizer()

            for i, doc in enumerate(data[:num_samples]):
                doc_out = []
                for wd in doc:
                    if wd not in stop_words:  # remove stopwords
                        doc_out.append(lemmatizer.lemmatize(wd))
                    else:
                        continue
                data_processed.append(doc_out)


            dct = corpora.Dictionary(data_processed)
            corpus = [dct.doc2bow(line) for line in data_processed]


            dct.save(self.dictName)  # save dict to disk
            MmCorpus.serialize(self.corpusName, corpus) # save corpus to disk
            return dct,corpus
        
        if self.load_cache:
            # try to load corpus, if fail it will try to re-build the corpus
            try:
                dct = corpora.Dictionary().load(self.dictName)
                corpus = MmCorpus(self.corpusName)
            except FileNotFoundError:
                print('try re-build dictionary and corpus from code')
                return build_corpus()

        return build_corpus()
        

        
