from cleaning import database_cleaner

import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

punctuation_ = set(string.punctuation)
stopwords_ = set(stopwords.words('english'))

def filter_tokens(sent):
    return([w for w in sent if not w in stopwords_ and not w in punctuation_])

def feature_matrix(corpus, tf_idf=True, stem_lem=None, ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None):
    '''
    Returns tf-idf vector for a given corpus (list/array of documents) by default
    Count vector(bag of words) returned when tf_idf is set to False
    Option to include stemming or lemmatizing
    '''
    tokens = [word_tokenize(doc) for doc in corpus]
    tokens_lower = [[word.lower() for word in sent] for sent in tokens]
    tokens_filtered = list(map(filter_tokens, tokens_lower))

    # Stemming-Lemmatizing
    if stem_lem == 'stem':
        stemmer_porter = PorterStemmer()
        tokens_filtered = [' '. join(list(map(stemmer_porter.stem, sent))) for sent in tokens_filtered]


    elif stem_lem == 'lem':
        lemmatizer = WordNetLemmatizer()
        tokens_filtered = [' '.join(list(map(lemmatizer.lemmatize, sent))) for sent in tokens_filtered]

    else:
        tokens_filtered = [' '.join(sent) for sent in tokens_filtered]

    # Vectorizing
    if tf_idf:
        vectorizer = TfidfVectorizer(stop_words=stopwords_, ngram_range=ngram_range,
                                     max_df=max_df, min_df=min_df, max_features=max_features)
        matrix = vectorizer.fit_transform(tokens_filtered) # sparse matrix

    else:
        vectorizer = CountVectorizer(stop_words=stopwords_, ngram_range=ngram_range,
                                     max_df=max_df, min_df=min_df, max_features=max_features)
        matrix = vectorizer.fit_transform(tokens_filtered) # sparse matrix

    return vectorizer, matrix
