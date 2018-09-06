from clustering_model import MyModel
from topic_model import MyTopicModel
from gensim_model import MyGenSimModel
from nlp_pipeline import clean_text

from flask import Flask, request, render_template, abort, Markup

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg') # to prevent plt image popups 

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

# LDA topic model and vectorizer(using sklearn)
with open('../data/pickle/pge_sklearn_LDA.pkl', 'rb') as f:
        sklearn_LDA = pickle.load(f)

with open('../data/pickle/pge_sklearn_LDA_vectorizer.pkl', 'rb') as f:
    sklearn_LDA_vectorizer = pickle.load(f)

# gensim topic models
with open('../data/pickle/pge_gensim_LDA.pkl', 'rb') as f:
        gensim_LDA = pickle.load(f)

with open('../data/pickle/pge_gensim_LDAMallet.pkl', 'rb') as f:
        gensim_LDAMallet = pickle.load(f)

# corresponding databases
majors_df = pd.read_json('../data/json/majors_database.json')
final_cluster_df = pd.read_json('../data/json/final_database.json')
final_NMF_df = pd.read_json('../data/json/final_NMF_database.json')
final_sklearn_LDA_df = pd.read_json('../data/json/final_sklearn_database_LDA.json')
final_gensim_LDA_df = pd.read_json('../data/json/final_gensim_database_LDA.json')
final_gensim_LDAMallet_df = pd.read_json('../data/json/final_gensim_database_LDAMallet.json')

# Choice of the model to be used, valid choices are 'LDA', 'LDAMallet', 'KMeans', 'NMF'
model_choice='LDAMallet'

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
    display_data = _get_model_results(search_text, top_n=10)[:5] # taking top-5
    return display_data.to_json(force_ascii=True,  orient='records')

# Each professor's page
@app.route('/professor/<int:professor_id>', methods=['GET'])
def professor(professor_id):
    prof_info = _get_prof_info(professor_id)
    prof_topic_distribution_plot = _get_prof_topic_distribution_plot(professor_id)
    return render_template('professor.html', prof_info=prof_info, prof_topic_distribution_plot=prof_topic_distribution_plot)

# Topics for all the professors
@app.route('/professor-topics.html', methods=['GET'])
def professor_topics():
    prof_info_df = _get_prof_info_df()
    prof_names = prof_info_df["faculty_name"].tolist()
    prof_ids = prof_info_df["id"].tolist()
    return render_template('professor-topics.html', prof_ids_and_names=dict(zip(prof_ids, prof_names)))

# Topics for a single professor
@app.route('/professor-topics/<int:professor_id>', methods=['POST'])
def professor_topic(professor_id):
    return _get_prof_topic_distribution_plot(professor_id)

# Get the professor's topic distribution plot
def _get_prof_topic_distribution_plot(professor_id):
    if professor_id > 87 or professor_id < 0:
        abort(404)
    else:
        prof_topic_distribution_plot = ""
        with open(f'static/plots/prof_topic_plots/tp_{professor_id}.html', 'r', encoding='utf-8') as plots_file:
            content = plots_file.read()
            # remove html, head and body tags
            prof_topic_distribution_plot = Markup(content.replace('<html><head><meta charset="utf-8" /></head><body>', '').replace('</body></html>', ''))
        
        return prof_topic_distribution_plot

# Get the information of a single professor 
def _get_prof_info(professor_id):
    '''
    Parameters
    ----------
    professor_id: The ID of the professor in majors database
    '''
    prof_info_df = _get_prof_info_df()
    return prof_info_df[prof_info_df["prof_id"] == professor_id].to_dict(orient='records')

# Choose database for lookup
def _get_prof_info_df():
    """Chooses the database for looking up professor information."""
    if model_choice == 'KMeans':
        prof_info_df = final_cluster_df

    elif model_choice == 'NMF':
        prof_info_df = final_NMF_df

    elif model_choice == 'LDA':
        prof_info_df = final_gensim_LDA_df
    
    elif model_choice == 'LDAMallet':
        prof_info_df = final_gensim_LDAMallet_df

    return prof_info_df

# Search and Ranking algorithm
def _get_model_results(search_text, top_n=5):
    '''
    Parameters
    ----------
    search_text: The text to be used for searching, could be research area or body of a paper abstract.
    top_n: Number of similar professors to be returned.
    '''
    
    if model_choice == 'KMeans':
        similarities = cluster_model.most_similar(search_text, cluster_vectorizer, top_n=top_n) # document_id, similarity
        search_df = _get_search_df(similarities, final_cluster_df)

    elif model_choice == 'NMF':
        similarities = NMF_model.most_similar(search_text, NMF_vectorizer, top_n=top_n) 
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
    search_df = _ranking_algo(results_df[results_df['paper_count'] > 20], weights=[0.05, 0.10, 0.15, 0.65]) # filter out professors with less than 10 papers in database
    return search_df

def _ranking_algo(results_df, weights=[0.05, 0.35, 0.15, 0.45]):
    """Rank search results based on university rank, number of papers in majors_database, 
    h-index, and similarity between the search_text and the corpus for the model."""
    search_df = results_df.copy()
    search_df["composite_score"] = weights[0] * search_df["rank"] + weights[1] * search_df["paper_count"] +  weights[2] * search_df["h_index"] + weights[3] * search_df["similarity"] 
    return search_df.sort_values(by="composite_score", ascending=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
