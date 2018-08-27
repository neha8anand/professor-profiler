from collections import Counter
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

import requests
from topic_model import MyTopicModel
from bson import json_util, ObjectId
import json
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from io import BytesIO
import random

app = Flask(__name__,
        static_url_path='')

# topic model and vectorizer
with open('../data/pge_topic_model.pkl', 'rb') as f:
        topic_model = pickle.load(f)

with open('../data/pge_topic_vectorizer.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

# Form page to submit text
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', data=display_data)

# My professor profiler app
@app.route('/submit', methods=['POST'])
def submit():
    user_data = request.json
    search_text = user_data["text_input"]
    display_data = get_model_results(search_text)
    return display_data.to_json(force_ascii=True, orient='records', default_handler=str)

# Search algorithm and Ranking algorithm
def get_model_results(search_text):
    '''
    Parameters
    ----------
    search_text: The text to be used for searching, could be research area or body of a paper abstract.
    '''
    similarities = topic_model.most_similar(search_text, topic_vectorizer, top_n=5) # document_id, similarity
    similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
    document_ids = list(map(int, similarities[:,0]))
    results_df = final_topic_df[final_topic_df.index.isin(document_ids)].sort_index()
    results_df['similarity'] = similarities[:,1]
    cols = ['faculty_name', 'faculty_title', 'research_areas', 'predicted_research_areas', 'office', 'email', 'phone', 'page', 'google_scholar_link']
    return results_df.sort_values(by='similarity')[cols]

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
    app.run(host='0.0.0.0', port=8080, debug=True)
