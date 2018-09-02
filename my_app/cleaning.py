import numpy as np
import pandas as pd

def database_cleaner(filename):
    '''
    Cleans the majors database json file (majors_database.json), extracts information about 
    top - 4 petroleum schools and returns a pandas dataframe for use in nlp pipeline.
    '''
    majors_df = pd.read_json(filename)

    # Extracting top-4 petroleum school information
    pge_schools = ['University of Texas--Austin (Cockrell)', 'Stanford University',
            'Texas A&M University--College Station', 'University of Tulsa']
    pge_schools_df = pd.DataFrame(majors_df['petroleum-engineering'][pge_schools])
    pge_schools_df.reset_index(level=0, inplace=True)

    
    # Expanding faculty_info column
    pge_df = pge_schools_df["petroleum-engineering"].apply(pd.Series)
    df_rearranged = pd.concat([pge_schools_df, pge_df], axis=1).drop("petroleum-engineering", axis=1)
    df_rearranged.rename(index=str, columns={"faculty_names": "faculty_info", 
                                             "index": "university_name"}, inplace=True)
    # Rearranging and cleaning df
    df_list = []

    for index, row in df_rearranged.iterrows():
        faculty_df = pd.DataFrame(row["faculty_info"]).T
        faculty_df["faculty_name"] = faculty_df.index
        faculty_df.reset_index(drop=True, inplace=True)
        
        univ_df = pd.DataFrame(row.loc[["university_name", "location", "rank", "score"]]).T 
        univ_dfs_rep = pd.concat([univ_df] * len(faculty_df), ignore_index=True)
        res_df = pd.concat([univ_dfs_rep, faculty_df], axis=1)
        
        df_list.append(res_df)
    
    df_final = df_list[0].append(df_list[1], ignore_index=True).append(df_list[2], ignore_index=True).append(df_list[3], ignore_index=True)

    # Extracting titles, abstracts from the papers column and stringing them together for a single faculty
    faculty_paper_titles_combined = []
    faculty_abstracts_combined = []
    faculty_abstracts_counts = []

    for index, row in df_final.iterrows():
        paper_titles = ''
        paper_abstracts = ''
        paper_abstracts_counts = 0

        for entry in row['papers'].values():
            if 'title' in entry:
                paper_titles += ' ' + entry['title']

            if 'abstract' in entry:
                paper_abstracts += ' ' + entry['abstract']
                paper_abstracts_counts += 1


        faculty_paper_titles_combined.append(paper_titles)
        faculty_abstracts_combined.append(paper_abstracts)
        faculty_abstracts_counts.append(paper_abstracts_counts)

    df_final.drop(columns=['papers'], inplace=True)
    df_final['paper_titles'] = faculty_paper_titles_combined
    df_final['abstracts'] = faculty_abstracts_combined
    df_final['paper_count'] = faculty_abstracts_counts

    # Make h_index column
    df_final["h_index"] = df_final["indices"].apply(pd.Series)["h-index"].apply(pd.Series)["All"]

    # Add prof_id column
    df_final["prof_id"] = df_final.index

    return df_final
