from cleaning import database_cleaner
from model import get_data, MyModel
import pandas as pd
import pickle

# pge_model
with open('../data/pge_model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('../data/pge_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

final_df = pd.read_json('../data/final_database.json')

# Search algorithm
def search(search_text, type='text_input'):
    '''
    type: 'text_input' for abstract/whole body of their past paper,
    'research_area_input' for research_area.

    search_text: the text to be used for searching.
    '''
    # If searching by research_area
    # If searching by text_input
    X_test = vectorizer.transform(search_text)
    y_test = model.predict(X_test) # predicted cluster label for given text
    results_df = final_df[final_df['predicted_cluster_num'] == y_test]
    return results_df[['faculty_name', 'research_areas', 'predicted_research_areas']]

# Ranking algorithm
if __name__ == '__main__':
    search_text = 'The most frequent problem encountered in day to day oil well production is the deposition of paraffin or wax inside the production tubing. The solution to this problem is frequent scraping and hot oil operation with help of Coiled Tubing Unit (CTU) thereby increasing the non productive time. The following paper proposes an alternative technique for dewaxing of production tubing.'
    results = search(search_text, type='text_input')
    print(results)
