"""
Module containing model fitting code for a web application that implements a
research interests identification topic model(Choice between LDA, NMF and LSI).
When run as a module, this will load a json dataset, train a decomposition
model using sklearn, and then pickle the resulting model object to disk.
"""
from cleaning import database_cleaner
from combine_databases import add_database
from nlp_pipeline import feature_matrix

import numpy as np
import pickle
import pandas as pd

from gensim import models, corpora

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

if __name__ == '__main__':
    # Create pge_database with updated predicted_research_areas based on top-10 features
    current_db_path = '../data/ut_database.json'
    new_db_paths = ['../data/stanford_database.json', '../data/tamu_database.json']
    combined_db_path = '../data/pge_database.json'
    add_database(current_db_path, new_db_paths, combined_db_path)

    corpus = get_corpus('../data/pge_database.json')
    # words occurring in only one document or in at least 80% of the documents are removed.
    vectorizer, matrix = vectorize_corpus(corpus, tf_idf=False, stem_lem=None, ngram_range=(1,1),
                                    max_df=0.8, min_df=5, max_features=None)
