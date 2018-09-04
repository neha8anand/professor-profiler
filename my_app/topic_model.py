"""
Module containing model fitting code for a web application that implements a
research interests identification topic model(Choice between LDA, NMF and LSI).
When run as a module, this will load a json dataset, train a decomposition
model using sklearn, and then pickle the resulting model object to disk.
"""
from cleaning import database_cleaner
from nlp_pipeline import feature_matrix, clean_text, add_stopwords

import numpy as np
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn

from nltk.corpus import stopwords
stopwords_ = stopwords.words('english')
more_stopwords_ = add_stopwords()
stopwords_.extend(more_stopwords_)

class MyTopicModel():
    """A topic model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a topic model to the resulting features.
    """

    def __init__(self, n_topics=12, algorithm='NMF'):

        if algorithm == 'LDA':
            # Build a Latent Dirichlet Allocation Model
            self._model = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online')
        elif algorithm == 'NMF':
            # Build a Non-Negative Matrix Factorization Model
            self._model = NMF(n_components=n_topics)
        elif algorithm == 'LSI':
            # Build a Latent Semantic Indexing Model
            self._model = TruncatedSVD(n_components=n_topics)

    def fit_transform(self, X):
        """Return transformed training data."""
        self.X = X
        self.transformed_X = self._model.fit_transform(X)
        return self.transformed_X

    def transform(self, X):
        """Return transformed new data."""
        return self._model.transform(X)

    def top_n_features(self, vectorizer, top_n=10):
        """Returns top n features/words for all topics given a vectorizer object."""
        topic_words = []
        for _, topic in enumerate(self._model.components_):
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        return np.array(topic_words)

    def most_similar(self, search_text, vectorizer, top_n=5):
        """Returns most similar professors for a given search text (cleaned and tokenized)."""
        x = self._model.transform(vectorizer.transform(clean_text(search_text,stopwords_)))[0]
        similarities = cosine_similarity(x.reshape(1, -1), self.transformed_X)
        pairs = enumerate(similarities[0])
        most_similar = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]
        return np.array(most_similar)

    def visualize_lda_model(self, matrix, vectorizer, **kwargs):
        """Visualize LDA model using pyLDAvis"""
        vis = pyLDAvis.sklearn.prepare(self._model, matrix, vectorizer, **kwargs)
        return vis

def get_corpus(filename):
    """Load raw data from a file and return vectorizer and feature_matrix.
    Parameters
    ----------
    filename: The path to a json file containing the university database.

    Returns
    -------
    corpus: A numpy array containing abstracts and titles.
    """
    df_cleaned = database_cleaner(filename)

    # For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
    df_filtered = df_cleaned[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
    missing = df_filtered['paper_titles'] == ''
    df_nlp = df_filtered[~missing]

    # Choosing abstracts and paper_titles to predict topics for a professor
    df_nlp['research_areas'] = df_nlp['research_areas'].apply(lambda x: " ".join(x))
    corpus = (df_nlp['paper_titles'] + df_nlp['abstracts'] + df_nlp['research_areas']).values

    return corpus

def vectorize_corpus(corpus, tf_idf=True, stem_lem=None, **kwargs):
    """
    Parameters
    ----------
    corpus: A numpy array containing abstracts and titles.

    Returns
    -------
    vectorizer: A numpy array containing TfidfVectorizer or CountVectorizer object.
    matrix: The feature matrix (numpy 2-D array) returned by the vectorizer object.
    """

    vectorizer, matrix = feature_matrix(corpus, tf_idf=True, stem_lem=None, **kwargs)

    return vectorizer, matrix


if __name__ == '__main__':
    corpus = get_corpus('../data/json/majors_database.json')
    # LDA
    lda_vectorizer, lda_matrix = vectorize_corpus(corpus, tf_idf=False, stem_lem=None, ngram_range=(1,1),
                                    max_df=0.8, min_df=5, max_features=None)
    lda_model = MyTopicModel(n_topics=12, algorithm='LDA')
    lda_pred = lda_model.fit_transform(lda_matrix)
    lda_topic_words = lda_model.top_n_features(lda_vectorizer, top_n=10)

    # NMF
    nmf_vectorizer, nmf_matrix = vectorize_corpus(corpus, tf_idf=True, stem_lem=None, ngram_range=(1,1),
                                    max_df=0.8, min_df=5, max_features=None)
    nmf_model = MyTopicModel(n_topics=12, algorithm='NMF')
    nmf_pred = nmf_model.fit_transform(nmf_matrix)
    nmf_topic_words = nmf_model.top_n_features(nmf_vectorizer, top_n=10)

    # Update database with predictions
    pge_df = database_cleaner('../data/json/majors_database.json')

    pge_df_updated_LDA = pge_df.copy()
    pge_df_updated_LDA['predicted_topic_num'] = [num[-1] for num in lda_pred.argsort(axis=1)]
    pge_df_updated_LDA['predicted_research_areas'] = [lda_topic_words[topic_num] for topic_num in pge_df_updated_LDA['predicted_topic_num']]
    pge_df_updated_LDA.to_json(path_or_buf='../data/json/final_sklearn_database_LDA.json')

    pge_df_updated_NMF = pge_df.copy()
    pge_df_updated_NMF['predicted_topic_num'] = [num[-1] for num in nmf_pred.argsort(axis=1)]
    pge_df_updated_NMF['predicted_research_areas'] = [nmf_topic_words[topic_num] for topic_num in pge_df_updated_NMF['predicted_topic_num']]
    pge_df_updated_NMF.to_json(path_or_buf='../data/json/final_sklearn_database_NMF.json')

    # Pickle models and vectorizers
    with open('../data/pickle/pge_sklearn_LDA.pkl', 'wb') as f:
        pickle.dump(lda_model, f)

    with open('../data/pickle/pge_sklearn_LDA_vectorizer.pkl', 'wb') as f:
        pickle.dump(lda_vectorizer, f)
    
    with open('../data/pickle/pge_sklearn_NMF.pkl', 'wb') as f:
        pickle.dump(nmf_model, f)

    with open('../data/pickle/pge_sklearn_NMF_vectorizer.pkl', 'wb') as f:
        pickle.dump(nmf_vectorizer, f)

