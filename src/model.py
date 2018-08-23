"""
Module containing model fitting code for a web application that implements a
research interests identification clustering model.
When run as a module, this will load a json dataset, train a clustering
model, and then pickle the resulting model object to disk.
"""
from cleaning import database_cleaner
from nlp_pipeline import feature_matrix

import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans

class MyModel():
    """A clustering model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a K-Means clustering model to the resulting features.
    """

    def __init__(self, n):
        self._clusterer = KMeans(n)

    def fit(self, X, y):
        """Fit a clustering model.
        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        Returns
        -------
        self: The fit model object.
        """
        # Code to fit the model.
        self._clusterer.fit(X)
        return self

    def fit_predict(self, X):
        """Return cluster assignments for new data."""
        return self._clusterer.fit_predict(X)

    def top_n_features(self, vocabulary, n):
        """Returns top n features for a given vocabulary object (Eg. vectorizer.vocabulary_)."""
        reverse_vocab = reverse_vocabulary(vocabulary)
        centroids = self._clusterer.cluster_centers_ # topics/research areas Kmeans has discovered
        indices = np.argsort(centroids, axis=1)
        top_n_indices = indices[:, -n:]
        top_n_features = np.array([reverse_vocab[index] for row in top_n_indices for index in row])
        top_n_features = top_n_features.reshape(len(centroids), -1) # topics with the top n greatest representation in each of the centroids
        return top_n_features

def get_data(filename):
    """Load raw data from a file and return vectorizer and feature_matrix.
    Parameters
    ----------
    filename: The path to a json file containing the university database.
    Returns
    -------
    vectorizer: A numpy array containing TfidfVectorizer or CountVectorizer object.
    matrix: A numpy array containing the feature matrix returned by the vectorizer object.
    """
    df_cleaned = database_cleaner(filename)

    # For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
    df_filtered = df_cleaned[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
    missing = df_filtered['paper_titles'] == ''
    num_missing = sum(missing)
    print(f'{num_missing} faculties have missing papers in {filename}')
    print('Running nlp-pipeline on faculties with non-missing papers...')

    df_nlp = df_filtered[~missing]

    # Choosing abstracts to predict topics for a professor
    corpus = df_nlp['abstracts'].values
    #corpus = df_nlp['paper_titles'].values
    vectorizer, matrix = feature_matrix(corpus, tf_idf=True, stem_lem=None, ngram_range=(1,1),
                                    max_df=0.8, min_df=2, max_features=None)

    return vectorizer, matrix

def reverse_vocabulary(vocabulary):
    """Reverses the vocabulary dictionary as returned by the vectorizer."""
    reverse_vocab = {}
    for key, value in vocabulary.items():
        reverse_vocab[value] = key
    return reverse_vocab

if __name__ == '__main__':
    # Create pge_database
    current_db_path = '../data/ut_database.json'
    new_db_paths = ['../data/stanford_database.json', '../data/tamu_database.json']
    combined_db_path = '../data/pge_database.json'
    add_database(current_db_path, new_db_paths, combined_db_path)
    
    vectorizer, matrix = get_data('../data/pge_database.json')
    model = MyModel(12)
    y_pred = model.fit_predict(matrix)
    # print(y_pred)

    with open('../data/pge_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../data/pge_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
