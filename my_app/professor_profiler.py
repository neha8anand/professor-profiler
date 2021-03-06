from nlp_pipeline import clean_text
from clustering_model import MyModel
from topic_model import MyTopicModel
import pandas as pd
import pickle

# cluster_model
with open('../data/pickle/pge_model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('../data/pickle/pge_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# topic_model
with open('../data/pickle/pge_topic_model.pkl', 'rb') as f:
        topic_model = pickle.load(f)

with open('../data/pickle/pge_topic_vectorizer.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

final_df = pd.read_json('../data/json/final_database.json')
final_topic_df = pd.read_json('../data/json/final_topic_database.json')

# Search algorithm and Ranking algorithm
def search(search_text, type='text_input', choose_model='topic_model'):
    '''
    Parameters
    ----------
    search_text: the text to be used for searching.
    type: 'text_input' for abstract/whole body of their past paper,
    'research_area_input' for research_area.
    model: 'topic_model' for topic modeling and 'cluster_model' for KMeans clustering
    '''
    # If searching by research_area
    # If searching by text_input
    if type == 'text_input':
        if choose_model == 'cluster_model':
            y_test = model.predict(vectorizer.transform(clean_text(search_text))) # predicted cluster label for given text
            # results_df = final_df[final_df['predicted_cluster_num'] == y_test[0]]
            similarities = model.most_similar(search_text, vectorizer, top_n=5) # document_id, similarity
            similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
            document_ids = list(map(int, similarities[:,0]))
            results_df = final_df[final_df.index.isin(document_ids)].sort_index()
            results_df['similarity'] = similarities[:,1]
            return results_df.sort_values(by='similarity')[['faculty_name', 'research_areas', 'predicted_research_areas']]
        else:
            similarities = topic_model.most_similar(clean_text(search_text), topic_vectorizer, top_n=5) # document_id, similarity
            similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
            document_ids = list(map(int, similarities[:,0]))
            results_df = final_topic_df[final_topic_df.index.isin(document_ids)].sort_index()
            results_df['similarity'] = similarities[:,1]
            cols = ['faculty_name', 'faculty_title', 'research_areas', 'predicted_research_areas', 'office', 'email', 'phone', 'page', 'google_scholar_link']
            search_df = results_df.sort_values(by='similarity')[cols][:5]
            return search_df

if __name__ == '__main__':
    search_text = 'sagd'
    results = search(search_text, type='text_input', choose_model='topic_model')
    print(results)
