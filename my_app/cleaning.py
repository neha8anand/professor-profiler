import numpy as np
import pandas as pd

def database_cleaner(filename):
    '''
    Cleans the university database json files (For example: tamu_database.json)
    and returns a pandas dataframe for use in nlp pipeline
    '''
    df = pd.read_json(filename)

    # Rearranging and cleaning df
    df_rearranged = df.copy()
    df_rearranged.drop(columns='university_name', inplace=True)
    df_rearranged.rename(index=str, columns={"faculty_names": "faculty_info"}, inplace=True)
    df_rearranged["faculty_name"] = df_rearranged.index
    df_rearranged.reset_index(drop=True, inplace=True)

    # Expanding faculty_info column
    faculty_info_df = df_rearranged["faculty_info"].apply(pd.Series)
    df_final = pd.concat([df_rearranged, faculty_info_df], axis=1).drop("faculty_info", axis=1)
    df_final.rename(index=str, columns={"title": "faculty_title"}, inplace=True)

    # Extracting titles, abstracts from the papers column and research_areas
    # and stringing them together for a single faculty
    faculty_paper_titles_combined = []
    faculty_abstracts_combined = []
    research_areas_combined = []

    for index, row in df_final.iterrows():
        paper_titles = ''
        paper_abstracts = ''

        for entry in row['papers'].values():
            if 'title' in entry:
                paper_titles += ' ' + entry['title']

            if 'abstract' in entry:
                paper_abstracts += ' ' + entry['abstract']


        faculty_paper_titles_combined.append(paper_titles)
        faculty_abstracts_combined.append(paper_abstracts)
        research_areas_combined.append(' '.join(x for x in row['research_areas']))

    df_final.drop(columns=['papers', 'research_areas'], inplace=True)
    df_final['paper_titles'] = faculty_paper_titles_combined
    df_final['abstracts'] = faculty_abstracts_combined
    df_final['research_areas'] = research_areas_combined

    return df_final
