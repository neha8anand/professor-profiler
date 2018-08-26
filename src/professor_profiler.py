from cleaning import database_cleaner
from clustering_model import get_corpus, vectorize_corpus, MyModel
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

# Search algorithm
def search(search_text, type='text_input', model='topic_model'):
    '''
    Parameters
    ----------
    search_text: the text to be used for searching.
    type: 'text_input' for abstract/whole body of their past paper,
    'research_area_input' for research_area.
    model: 'topic_model' for soft clustering and 'cluster_model' for hard clustering
    '''
    # If searching by research_area
    # If searching by text_input
    if type == 'text_input':
        if model == 'cluster_model':
            X_test = vectorizer.transform(search_text)
            y_test = model.predict(X_test) # predicted cluster label for given text
            results_df = final_df[final_df['predicted_cluster_num'] == y_test[0]]
        else:
            x = topic_model.transform(topic_vectorizer.transform([text]))[0]
            similarities = most_similar(x, nmf_Z)
            for (document_id, similarity) in similarities:
                print(final_df.iloc[document_id].loc[['faculty_name', 'research_areas']])
    return results_df[['faculty_name', 'research_areas', 'predicted_research_areas']]

# Ranking algorithm
if __name__ == '__main__':
    search_text = ['For wells drilled in shale gas reservoirs to be economic, hydraulic fracturing has become a common completion practice. Production from these completed wells is highly dependent on the characteristics of proppants placed in the created fractures. Fracturing leads to an interaction between the minerals of the proppants, formation and the fluids; this results in the phenomenon of proppants diagenesis. It involves mechanisms such as diffusion, dissolution, precipitation along with chemical reactions that take place at the fracture surface. Over time, this combined process results in a loss of proppant pack permeability thereby leading to a decline in well productivity. This occurs due to changes in compositional differences between proppants, fracturing fluid and the formation. A mathematical model, representing this phenomenon is developed in this paper']
    results = search(search_text, type='text_input')
    print(results)
