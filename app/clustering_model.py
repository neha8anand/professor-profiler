"""
Module containing model fitting code for a web application that implements a
research interests identification K-Means clustering model.
When run as a module, this will load a json dataset, train a clustering
model, and then pickle the resulting model object to disk.
"""
from cleaning import database_cleaner
from combine_databases import add_database
from nlp_pipeline import feature_matrix

import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class MyModel():
    """A clustering model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a K-Means clustering model to the resulting features.
    """

    def __init__(self, n):
        self._clusterer = KMeans(n)

    def fit_predict(self, X):
        """Return cluster assignments for training data.
        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        Returns
        -------
        y: The fit model predictions.
        """
        self.X = X
        return self._clusterer.fit_predict(X)

    def predict(self, X):
        """Return cluster assignments for new data."""
        return self._clusterer.predict(X)

    def top_n_features(self, vocabulary, n):
        """Returns top n features for a given vocabulary object (Eg. vectorizer.vocabulary_)."""
        reverse_vocab = reverse_vocabulary(vocabulary)
        centroids = self._clusterer.cluster_centers_ # topics/research areas Kmeans has discovered
        indices = np.argsort(centroids, axis=1)
        top_n_indices = indices[:, -n:]
        top_n_features = np.array([reverse_vocab[index] for row in top_n_indices for index in row])
        top_n_features = top_n_features.reshape(len(centroids), -1) # topics with the top n greatest representation in each of the centroids
        return top_n_features

    def most_similar(self, search_text, vectorizer, top_n=5):
        """Returns top n most similar professors for a given search text."""
        x = (vectorizer.transform([search_text]))
        dists = euclidean_distances(x, self.X)
        pairs = enumerate(dists[0])
        most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
        return np.array(most_similar)

def get_corpus(filename):
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
    corpus = (df_nlp['paper_titles'] + df_nlp['abstracts']).values

    return corpus

def vectorize_corpus(corpus, tf_idf=True, stem_lem=None, **kwargs):
    """
    Parameters
    ----------
    corpus: A numpy array containing abstracts.
    Returns
    -------
    vectorizer: A numpy array containing TfidfVectorizer or CountVectorizer object.
    matrix: The feature matrix (numpy 2-D array) returned by the vectorizer object.
    """

    vectorizer, matrix = feature_matrix(corpus, tf_idf=True, stem_lem=None, **kwargs)

    return vectorizer, matrix

def reverse_vocabulary(vocabulary):
    """Reverses the vocabulary dictionary as returned by the vectorizer."""
    reverse_vocab = {}
    for key, value in vocabulary.items():
        reverse_vocab[value] = key
    return reverse_vocab

if __name__ == '__main__':
    # Create pge_database with updated predicted_research_areas based on top-10 features
    current_db_path = '../data/ut_database.json'
    new_db_paths = ['../data/stanford_database.json', '../data/tamu_database.json', '../data/utulsa_database.json']
    combined_db_path = '../data/pge_database.json'
    add_database(current_db_path, new_db_paths, combined_db_path)

    corpus = get_corpus('../data/pge_database.json')
    # words occurring in only one document or in at least 80% of the documents are removed.
    vectorizer, matrix = vectorize_corpus(corpus, tf_idf=True, stem_lem=None, ngram_range=(1,1),
                                    max_df=0.8, min_df=2, max_features=None)
    model = MyModel(12)
    y_pred = model.fit_predict(matrix)

    pge_df = database_cleaner('../data/pge_database.json')
    top_ten_features = model.top_n_features(vectorizer.vocabulary_, 10)
    pge_df['predicted_cluster_num'] = y_pred
    pge_df['predicted_research_areas'] = [top_ten_features[num] for num in y_pred]
    pge_df.to_json(path_or_buf='../data/final_database.json')

    with open('../data/pge_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../data/pge_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)