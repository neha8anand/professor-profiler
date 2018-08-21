"""
Module containing model to combine new university databases to the existing database.
When run as a module, it creates the database with ut_pge as the base database.
"""
import json

def add_database(current_db_path, new_db_paths, combined_db_path):
    """
    Adds new university databases (json object) to the existing database(json object).

    Parameters
    ----------
    current_db_path: filepath of the current database (should be a .json file)
    new_db_path: list of filepaths of the new database to be added (should be a .json file)
    combined_db_path: filepath of the combined database, where the combined_db .json file should be stored.

    Returns
    -------
    None
    """

    with open(current_db_path) as fo:
        data1 = json.load(fo)

    for new_db_path in new_db_paths:

        with open(new_db_path) as fo:
            data2 = json.load(fo)

        data1.update(data2)

    with open(combined_db_path, "w") as fo:
        json.dump(data1, fo)
