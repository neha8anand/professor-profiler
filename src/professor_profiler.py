from cleaning import database_cleaner
from clustering_model import MyModel
from topic_model import MyTopicModel
import pandas as pd
import pickle

# cluster_model
with open('../data/pge_model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('../data/pge_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# topic_model
with open('../data/pge_topic_model.pkl', 'rb') as f:
        topic_model = pickle.load(f)

with open('../data/pge_topic_vectorizer.pkl', 'rb') as f:
    topic_vectorizer = pickle.load(f)

final_df = pd.read_json('../data/final_database.json')
final_topic_df = pd.read_json('../data/final_topic_database.json')

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
            y_test = model.predict(vectorizer.transform([search_text])) # predicted cluster label for given text
            # results_df = final_df[final_df['predicted_cluster_num'] == y_test[0]]
            similarities = model.most_similar(search_text, vectorizer, top_n=5) # document_id, similarity
            similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
            document_ids = list(map(int, similarities[:,0]))
            results_df = final_df[final_df.index.isin(document_ids)].sort_index()
            results_df['similarity'] = similarities[:,1]
            return results_df.sort_values(by='similarity')[['faculty_name', 'research_areas', 'predicted_research_areas']]
        else:
            similarities = topic_model.most_similar(search_text, topic_vectorizer, top_n=5) # document_id, similarity
            similarities = similarities[similarities[:,0].argsort()] # sorting by document_id
            document_ids = list(map(int, similarities[:,0]))
            results_df = final_topic_df[final_topic_df.index.isin(document_ids)].sort_index()
            results_df['similarity'] = similarities[:,1]
            return results_df.sort_values(by='similarity')[['faculty_name', 'research_areas', 'predicted_research_areas']]

if __name__ == '__main__':
    search_text = "Steam injection is a widely used oil-recovery method that has been commercially successful in many types of heavy-oil reservoirs, including the oil sands of Alberta, Canada. Steam is very effective in delivering heat that is the key to heavy-oil mobilization. In the distant past in California, and also recently in Alberta, solvents were/are being used as additives to steam for additional viscosity reduction. The current applications are in field projects involving steam-assisted gravity drainage (SAGD) and cyclic steam stimulation (CSS).The past and present projects using solvents alone or in combination with steam are reviewed and evaluated, including enhanced solvent SAGD (ES-SAGD) and liquid addition to steam for enhancing recovery (LASER). The use of solvent in other processes, such as effective solvent extraction incorporating electromagnetic heating (ESEIEH) and after cold-heavy-oil production with sand (CHOPS), are also reviewed. The theories behind the use of solvents with steam are outlined. These postulate additional heavy-oil/bitumen mobilization; oil mobilization ahead of the steam front; and oil mobilization by solvent dispersion caused by frontal instability. The plausibility of the different approaches and solvent availability and economics are also discussed."
    results = search(search_text, type='text_input', choose_model='topic_model')
    print(results)
