"""
Module containing visualization code for creating topic distribution plots for university professors 
and bubble plot indicating which professors belong to which dominant topic.
When run as a module, this will load a json dataset, create the visualizations and store 
the resulting plots to disk.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def create_visualization(filename):
    df = pd.read_json(filename)
    image_locations = []

    # Topic distribution plots
    for _, row in df.iterrows():
        _, ax = plt.subplots()
        perc_contributions = row["Perc_Contributions"]
        topic_num = pd.Series('Topic ' + str(num) for num in row["Dominant_Topics"])
        ax.barh(topic_num, perc_contributions, align='center')
        ax.set_yticks(topic_num)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Percent Contribution')
        ax.set_ylabel('Topic Numbers')
        ax.set_title(f'Topic distribution for {row["faculty_name"]}')
        image_location = f'plots/topic_plots/tp_{row["prof_id"]}'
        image_locations.append(image_location)
        plt.savefig(image_location)
        plt.close()
    
    df['image_locations'] = image_locations

    return df

if __name__ == '__main__':
    df_updated = create_visualization('../data/json/final_gensim_database_LDAMallet.json')
    df_updated.to_json(path_or_buf='../data/json/final_gensim_database_LDAMallet.json')