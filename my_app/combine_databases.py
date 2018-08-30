"""
Module containing model to combine new majors databases to the existing database.
"""
import pandas as pd

def add_database(current_db_path, new_db_paths, combined_db_path):
    """
    Adds new majors databases (json object) to the existing database(json object).

    Parameters
    ----------
    current_db_path: filepath of the current database (should be a .json file)
    new_db_paths: filepath of the new database to be added (should be a .json files)
    combined_db_path: filepath of the combined database, where the combined_db.json file should be stored.

    Returns
    -------
    None
    """
    current_df = pd.read_json(current_db_path)
    new_dfs = [pd.read_json(new_db_path) for new_db_path in new_db_paths]
    combined_df = pd.concat([current_df, *new_dfs])
    combined_df.to_json(path_or_buf=combined_db_path)
