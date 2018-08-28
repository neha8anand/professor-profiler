"""
Module containing model fitting code for a web application that implements a
research interests identification topic model using gensim(Choice between LDA, LSI and LDAMallet).
When run as a module, this will load a json dataset, train a decomposition
model using gensim, and then pickle the resulting model object to disk.
"""
from combine_databases import add_database
from nlp_pipeline import clean_text
from pyLDAvis_mallet import get_LDA_data

import numpy as np
import pickle
import pandas as pd

# Gensim
import gensim
from gensim.models import CoherenceModel
from gensim import models, corpora, similarities
from gensim.test.utils import datapath

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

# NLTK
from nltk.corpus import stopwords
from nltk import word_tokenize

# spacy for lemmatization
import spacy

import gzip
import os

mallet_path = '~/Documents/GitHub/capstone/mallet-2.0.8/bin/mallet' # update this path

class MyGenSimModel():
    """A gensim based topic model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a topic model to the resulting features.
    """

    def __init__(self, n_topics=9, algorithm='LDAMallet', tf_idf=True, bigrams=False, trigrams=False, lemmatization=False):
        self.n_topics = n_topics
        self.algorithm = algorithm
        self.tf_idf = tf_idf
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.lemmatization = lemmatization

    def fit_transform(self, data):
        """Return transformed training data."""
        # For gensim we need to tokenize the data and filter out stopwords
        self.tokens_filtered = [clean_text(doc) for doc in data]

        # Build a Dictionary - association word to numeric id
        self.dictionary = corpora.Dictionary(self.tokens_filtered)

        # Transform the collection of texts to a numerical form [(word_id, count), ...]
        self.corpus = [dictionary.doc2bow(text) for text in self.tokens_filtered]

        # tf-idf vectorizer
        if tf_idf:
            self._tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
            self.corpus = self._tfidf_model[self.corpus]

        if algorithm == 'LDA':
            # Build a Latent Dirichlet Allocation Model
            self._model = models.ldamodel.LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)

        elif algorithm == 'LDAMallet':
            # Build a Mallet Model
            self._model = ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=n_topics, id2word=dictionary, prefix='~/Documents/Github/capstone/')

        elif algorithm == 'LSI':
            # Build a Latent Semantic Indexing Model
            self._model = models.LsiModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)


        self.transformed_X = self._model.fit_transform(X)
        return self.transformed_X

    def transform(self, search_text):
        """Return transformed new data."""
        bow = self.dictionary.doc2bow(clean_text(search_text))

        if self.tf_idf:
            return self.model[self.tfidf_model[bow]])
        return self.model[bow]

    def perplexity(self):
        """Returns perplexity for LDA model. Measures per-word likelihood bound, using a chunk of documents as evaluation corpus."""
        return self.model.log_perplexity(self.corpus)

    def coherence_score(self):
        """Returns topic coherence for topic models. This is the implementation of the four stage topic coherence pipeline."""
        coherence_model = CoherenceModel(model=self.model, texts=self.tokens_filtered, dictionary=self.dictionary, coherence='c_v')
        return coherence_model.get_coherence()

    def most_similar(self, search_text, top_n=5):
        """Returns most similar professors for a given search text (cleaned and tokenized)."""
        lda_index = similarities.MatrixSimilarity(self.model[self.corpus])
        similarity_results = lda_index[self.transform(search_text)]
        similarity_results = sorted(enumerate(similarity_results), key=lambda item: -item[1])
        return similarity_results[:top_n]

    def visualize_lda_model(self):
        """ Visualize LDA model using pyLDAvis"""
        vis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary)
        return vis

    def visualize_lda_mallet(self):
        """ Visualize LDA model using pyLDAvis"""
        dataDir = "/Users/Neha/Documents/GitHub/capstone"
        statefile = 'state.mallet.gz'
        model = get_LDA_data(dataDir, statefile)
        vis = pyLDAvis.gensim.prepare(**model)
        return vis

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        return(sent_topics_df)

def get_data(filename):
    """Load raw data from a file and return vectorizer and feature_matrix.
    Parameters
    ----------
    filename: The path to a json file containing the university database.
    Returns
    -------
    corpus: A numpy array containing abstracts.
    """
    df_cleaned = database_cleaner(filename)

    # For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
    df_filtered = df_cleaned[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
    missing = df_filtered['paper_titles'] == ''
    num_missing = sum(missing)
    print(f'{num_missing} faculties have missing papers in {filename}')
    print('Running nlp-pipeline on faculties with non-missing papers...')

    df_nlp = df_filtered[~missing]

    # Choosing abstracts and paper_titles to predict topics for a professor
    data = (df_nlp['paper_titles'] + df_nlp['abstracts']).values

    return data

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


if __name__ == '__main__':
    # Create pge_database with updated predicted_research_areas based on top-10 features
    current_db_path = '../data/ut_database.json'
    new_db_paths = ['../data/stanford_database.json', '../data/tamu_database.json', '../data/utulsa_database.json']
    combined_db_path = '../data/pge_database.json'
    add_database(current_db_path, new_db_paths, combined_db_path)

    data = get_data('../data/pge_database.json')

    # Save model to disk.
    temp_file = datapath("model")
    lda.save(temp_file)

    # Load a potentially pretrained model from disk.
    lda = LdaModel.load(temp_file)
