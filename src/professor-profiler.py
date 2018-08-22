from combine_databases import add_database
from model import get_data, MyModel

# Create pge_database
current_db_path = '../data/ut_database.json'
new_db_paths = ['../data/stanford_database.json', '../data/tamu_database.json']
combined_db_path = '../data/pge_database.json'
add_database(current_db_path, new_db_paths, combined_db_path)

# pge_model
vectorizer, matrix = get_data(combined_db_path)
model = MyModel(10)
y_pred = model.fit_predict(matrix)
print(y_pred)
print(len(y_pred))
