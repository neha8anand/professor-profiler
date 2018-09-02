"""
Module containing code for comparing gensim based topic models (LDA, LSI, LDAMallet and HDP).
When run as a module, this will load a json dataset, train a decomposition
model using these models, optimize the number of topics for the best model, and then pickle
the resulting optimum model object to disk.
"""
from cleaning import database_cleaner
from gensim_model import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_optimum_model(data, num_topics):
    """Compare different LDA based models in gensim based on coherence score given a dataset and number of topics."""

    # Initiate models 
    # LDA
    lda_model = MyGenSimModel(num_topics=num_topics, algorithm='LDA', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False)
    # LDAMallet (tf_idf always false)
    ldamallet_model = MyGenSimModel(num_topics=num_topics, algorithm='LDAMallet', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False)
    # LSI
    lsi_model = MyGenSimModel(num_topics=num_topics, algorithm='LSI', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False)
    # HDP (num_topics not required)
    hdp_model = MyGenSimModel(algorithm='HDP', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False)
    
    model_list = [lda_model, ldamallet_model, lsi_model, hdp_model]

    return compare_models(data, model_list)

def evaluate_bar_graph(coherences, indices, title):
    """
    Function to plot bar graph comparing coherence scores.

    Parameters:
    -----------
    coherences: list of coherence scores
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Scores')
    plt.title(title)
    plt.savefig(title + '.png', bbox_inches='tight',)
    plt.close()

if __name__ == '__main__':
    data = get_data('../data/json/majors_database.json')

    # Step 1: Choose best model for this data
    coherence_scores, _ = get_optimum_model(data=data, num_topics=7)
    # model.transform(data)
    evaluate_bar_graph(coherence_scores,
                    ['LDA', 'LDAMallet', 'LSI', 'HDP'], title='Comparison between gensim topic models')

    # On the basis of visualization using pyLDAvis and coherence score, LDAMallet model seems the best.
    # Step 2: Choose optimum number of clusters for the LDAMallet model
    model = MyGenSimModel(num_topics=7, algorithm='LDAMallet', tf_idf=False, bigrams=False, trigrams=False, lemmatization=False)
    model.transform(data)
    start, limit, step = 5, 17, 2
    list_num_topics = np.array(range(start, limit, step))
    coherence_values = compute_coherence_values(dictionary=model.dictionary, corpus=model.corpus, texts=model.tokens, limit=limit, start=start, step=step, algorithm=model.algorithm)
    optimum_num_topics = list_num_topics[np.argmax(np.array(coherence_values))]

    # Generate coherence plot for the optimum model
    title = f'Coherence Plot for optimum {model.algorithm} model'
    coherence_plot(list_num_topics, coherence_values, title=title)

    # Step 3: Fit optimum model to training data
    optimum_model = MyGenSimModel(num_topics=optimum_num_topics, algorithm=model.algorithm, tf_idf=model.tf_idf, bigrams=model.bigrams, trigrams=model.trigrams, lemmatization=model.lemmatization)
    optimum_model.transform(data)
    optimum_model.fit()
    print(f'The optimum model parameters- algorithm: {optimum_model.algorithm}, num_topics: {optimum_model.num_topics}, tf_idf: {optimum_model.tf_idf}, bigrams: {optimum_model.bigrams}, trigrams: {optimum_model.trigrams}, lemmatization: {optimum_model.lemmatization}')
    print('\n')
    print(f'The optimum model coherence score is: {optimum_model.coherence_score()}')

    # Append to pge_database with updated predicted_research_areas based on top-10 features
    pge_df = database_cleaner('../data/json/majors_database.json')
    doc_topics_df = optimum_model.format_document_topics()
    pge_df_updated = pd.concat([pge_df, doc_topics_df], axis=1)
    pge_df_updated.to_json(path_or_buf='../data/json/final_gensim_database_optimum.json')

    # Save html for the pyLDAvis visualization of LDAMallet model
    vis = optimum_model.visualize_lda_mallet()
    pyLDAvis.save_html(data=vis, fileobj="templates/LDAMallet_optimum.html") 

    # Pickle model (has associated dictionary and tf_idf model)
    with open('../data/pickle/pge_gensim_optimum.pkl', 'wb') as f:
        pickle.dump(optimum_model, f)