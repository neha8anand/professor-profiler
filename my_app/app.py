from clustering_model import MyModel
from topic_model import MyTopicModel
from gensim_model import MyGenSimModel
from nlp_pipeline import clean_text

from flask import Flask, request, render_template

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg') # to prevent plt image popups

from bson import json_util, ObjectId
import json

from io import BytesIO

app = Flask(__name__,
        static_url_path='')

# K-Means clustering model and vectorizer(using sklearn)
with open('../data/pickle/pge_model.pkl', 'rb') as f:
        cluster_model = pickle.load(f)

with open('../data/pickle/pge_vectorizer.pkl', 'rb') as f:
    cluster_vectorizer = pickle.load(f)

# NMF topic model and vectorizer(using sklearn)
with open('../data/pickle/pge_NMF_model.pkl', 'rb') as f:
        NMF_model = pickle.load(f)

with open('../data/pickle/pge_NMF_vectorizer.pkl', 'rb') as f:
    NMF_vectorizer = pickle.load(f)

# NMF topic model and vectorizer(using sklearn)
with open('../data/pickle/pge_sklearn_LDA.pkl', 'rb') as f:
        sklearn_LDA = pickle.load(f)

with open('../data/pickle/pge_sklearn_LDA_vectorizer.pkl', 'rb') as f:
    sklearn_LDA_vectorizer = pickle.load(f)

# gensim model and vectorizer
with open('../data/pickle/pge_gensim_LDA.pkl', 'rb') as f:
        gensim_LDA = pickle.load(f)

with open('../data/pickle/pge_gensim_LDAMallet.pkl', 'rb') as f:
        gensim_LDAMallet = pickle.load(f)

# corresponding databases
final_cluster_df = pd.read_json('../data/json/final_database.json')
final_NMF_df = pd.read_json('../data/json/final_NMF_database.json')
final_sklearn_LDA_df = pd.read_json('../data/json/final_sklearn_database_LDA.json')
final_gensim_LDA_df = pd.read_json('../data/json/final_gensim_database_LDA.json')
final_gensim_LDAMallet_df = pd.read_json('../data/json/final_gensim_database_LDAMallet.json')

# Form page to submit text
@app.route('/', methods=['GET'])
@app.route('/index.html', methods=['GET'])
def index():
    return render_template('index.html')

# pyLDAvis html
@app.route('/visualize', methods=['GET'])
def visualize():
    return render_template('LDAMallet.html')
    
# My professor profiler app
@app.route('/submit', methods=['POST'])
def submit():
    user_data = request.json
    search_text = user_data["text_input"]
    display_data = _get_model_results(search_text, model_choice='NMF', top_n=10)[:5] # taking top-5
    return display_data.to_json(force_ascii=True,  orient='records')

# Search and Ranking algorithm
def _get_model_results(search_text, model_choice='LDAMallet', top_n=5):
    '''
    Parameters
    ----------
    search_text: The text to be used for searching, could be research area or body of a paper abstract.
    model_choice: Choice of the model to be used, valid choices are 'LDA', 'LDAMallet', 'KMeans', 'NMF'
    '''
    # list of columns to be displayed to the user
    # cols = ['faculty_name', 'university_name', 'rank', 'title', 'rating', 'tags', 'research_areas', 'location', 'office', 'email', 'phone', 'page', 'google_scholar_link', 'indices', 'citations']
    
    if model_choice == 'KMeans':
        # y_test = cluster_model.predict(cluster_vectorizer.transform(clean_text(search_text))) # predicted cluster label for given text
        # results_df = final_df[final_df['predicted_cluster_num'] == y_test[0]]
        similarities = cluster_model.most_similar(search_text, cluster_vectorizer, top_n=top_n) # document_id, similarity
        search_df = _get_search_df(similarities, final_cluster_df)

    elif model_choice == 'NMF':
        similarities = NMF_model.most_similar(search_text, NMF_vectorizer, top_n=top_n) # document_id, similarity
        search_df = _get_search_df(similarities, final_NMF_df)

    elif model_choice == 'LDA':
        similarities = gensim_LDA.most_similar(search_text, top_n=top_n)
        search_df = _get_search_df(similarities, final_gensim_LDA_df)
    
    elif model_choice == 'LDAMallet':
         similarities = gensim_LDAMallet.most_similar(search_text, top_n=top_n)
         search_df = _get_search_df(similarities, final_gensim_LDAMallet_df)

    return search_df

def _get_search_df(similarities, final_df):
    """Returns a dataframe containing information about similar professors."""
    similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
    document_ids = list(map(int, similarities[:,0]))
    df = final_df.copy()
    results_df = df[df.index.isin(document_ids)].sort_index()
    results_df['similarity'] = similarities[:,1]
    search_df = _ranking_algo(results_df[results_df['paper_count'] > 20], weights=[0.05, 0.10, 0.15, 0.65]) # filter people with less than 10 papers in database
    return search_df

def _ranking_algo(results_df, weights=[0.05, 0.35, 0.15, 0.45]):
    """Rank search results based on university rank, number of papers in majors_database, 
    h-index, and similarity between the search_text and the corpus for the model."""
    search_df = results_df.copy()
    search_df["composite_score"] = weights[0] * search_df["rank"] + weights[1] * search_df["paper_count"] +  weights[2] * search_df["h_index"] + weights[3] * search_df["similarity"] 
    return search_df.sort_values(by="composite_score", ascending=False)

# @app.route('/plot.png')
# def get_graph():
#     display_data = get_model_results()
#     df = pd.DataFrame(display_data)
#     plt.figure(figsize=(10,4))
#     ax = sns.countplot(data=df, x='country', palette='husl')
#     ax.set(xlabel='Event Country', ylabel='Number of Flagged Events', title='Number of Flagged Events by Country')
#     image = BytesIO()
#     plt.savefig(image)
#     plt.close()
#     return image.getvalue(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
