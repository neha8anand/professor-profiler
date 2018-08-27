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

with open('../data/pge_topic_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    display_data = get_model_results()

    return render_template('index.html', data=display_data)

@app.route('/submit', methods=['POST'])
def submit():
    display_data = get_model_results()
    json = display_data.to_json(force_ascii=True, orient='records', default_handler=str)
    return json

def get_model_results():
    test_df = get_live_data()

    return display_data

@app.route('/plot.png')
def get_graph():
    display_data = get_model_results()
    df = pd.DataFrame(display_data)
    plt.figure(figsize=(10,4))
    ax = sns.countplot(data=df, x='country', palette='husl')
    ax.set(xlabel='Event Country', ylabel='Number of Flagged Events', title='Number of Flagged Events by Country')
    image = BytesIO()
    plt.savefig(image)
    plt.close()
    return image.getvalue(), 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
