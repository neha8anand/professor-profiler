from combine_databases import add_database
from model import get_data, MyModel

# pge_model
with open('../data/pge_model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('../data/pge_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)



# Search algorithm
