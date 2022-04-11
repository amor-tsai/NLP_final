import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import spacy
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_corpus, common_dictionary

class LDA(object):
        # construct
    def __init__(self, filepath='', corpus='', dictionary='', toload=False, k_topic=5):
        self.k_topic = k_topic
        self.name = filepath
        self.corpus=corpus
        self.dictionary=dictionary
        if toload:
            self.model = LdaModel.load(self.name)
        else:
            self.model = LdaModel(self.corpus,id2word=self.dictionary,num_topics=self.k_topic)
    def save(self,filepath):
        if self.model:
            self.model.save(filepath)
            
# read corpus from csv
train = pd.read_csv(r"/users/wangp/work/mypython/train.csv")
# Set Column Names 
train.columns = ['ClassIndex', 'Title', 'Description']
# Combine Title and Description
# Because better accuracy than using them as separate features
X_train = train['Title'] + " " + train['Description'] 
doc_list = X_train[0:99].tolist() # only use 100 documents

print(doc_list[0:2])
# prepare data
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = stopwords.words('english')

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    [trigram_mod[bigram_mod[doc]] for doc in texts]

data_words = list(sent_to_words(doc_list))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_nostops = remove_stopwords(data_words)

data_words_bigrams = make_bigrams(data_words_nostops)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
   'NOUN', 'ADJ', 'VERB', 'ADV'
])

print(data_lemmatized[0:2])
#doc_tokenized = [simple_preprocess(doc) for doc in doc_list]
dictionary = corpora.Dictionary(data_lemmatized)
#BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
texts = data_lemmatized
BoW_corpus = [dictionary.doc2bow(text) for text in texts]

#id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
lda_model = LdaModel(
   corpus=BoW_corpus, id2word=dictionary, num_topics=6, #random_state=100, 
   update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True
)

#if need to save corpus
#import pickle
#with open("corpus_2.pkl", "wb") as f:
#    pickle.dump(BoW_corpus, f) 
#with open("corpus_2.pkl", "wb") as f:
#    pickle.dump(BoW_corpus, f)     

#lda_ml = LDA(corpus=BoW_corpus, dictionary=dictionary,k_topic=6)
lda_model.save('first_topic_model_2')
print('\nPerplexity: ', lda_model.log_perplexity(BoW_corpus))

# load topic model
lda_topic_model = LdaModel.load('first_topic_model_2')
print('\nPerplexity: ', lda_topic_model.log_perplexity(BoW_corpus))
