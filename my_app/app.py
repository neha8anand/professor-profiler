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
with open('../data/pickle/pge_topic_model.pkl', 'rb') as f:
        topic_model = pickle.load(f)

with open('../data/pickle/pge_topic_vectorizer.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

# gensim model and vectorizer
with open('../data/pickle/pge_gensim_LDA.pkl', 'rb') as f:
        gensim_LDA = pickle.load(f)

with open('../data/pickle/pge_gensim_LDAMallet.pkl', 'rb') as f:
        gensim_LDAMallet = pickle.load(f)

# corresponding databases
final_cluster_df = pd.read_json('../data/json/final_database.json')
final_topic_df = pd.read_json('../data/json/final_topic_database.json')
final_gensim_LDA_df = pd.read_json('../data/json/final_gensim_database_LDA.json')
final_gensim_LDAMallet_df = pd.read_json('../data/json/final_gensim_database_LDAMallet.json')

# Form page to submit text
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# My professor profiler app
@app.route('/submit', methods=['POST'])
def submit():
    user_data = request.json
    search_text = user_data["text_input"]
    display_data = _get_model_results(search_text, model_choice='NMF')
    return display_data.to_json(force_ascii=True,  orient='records')

# Search and Ranking algorithm
def _get_model_results(search_text, model_choice='LDAMallet'):
    '''
    Parameters
    ----------
    search_text: The text to be used for searching, could be research area or body of a paper abstract.
    model_choice: Choice of the model to be used, valid choices are 'LDA', 'LDAMallet', 'KMeans', 'NMF'
    '''
    # list of columns to be displayed to the user
    
    if model_choice == 'NMF':
        similarities = topic_model.most_similar(clean_text(search_text), topic_vectorizer, top_n=5) # document_id, similarity
        similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
        document_ids = list(map(int, similarities[:,0]))
        df = final_topic_df.copy()
        results_df = df[df.index.isin(document_ids)].sort_index()
        results_df['similarity'] = similarities[:,1]
        cols = ['faculty_name', 'university_name', 'rank', 'title', 'rating', 'tags', 'research_areas', 'location', 'office', 'email', 'phone', 'page', 'google_scholar_link', 'indices', 'citations']
        search_df = results_df.sort_values(by='similarity')[cols][:5]
    
    elif model_choice == 'KMeans':
        y_test = model.predict(vectorizer.transform(clean_text(search_text))) # predicted cluster label for given text
        # results_df = final_df[final_df['predicted_cluster_num'] == y_test[0]]
        similarities = model.most_similar(search_text, vectorizer, top_n=5) # document_id, similarity
        similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
        document_ids = list(map(int, similarities[:,0]))
        df = final_df.copy()
        results_df = df[df.index.isin(document_ids)].sort_index()
        results_df['similarity'] = similarities[:,1]
        search_df = results_df.sort_values(by='similarity')[['faculty_name', 'research_areas', 'predicted_research_areas']]

    elif model_choice == 'LDA':
        pass
    
    elif model_choice == 'LDAMallet':
        pass

    return search_df

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
