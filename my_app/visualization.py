"""
Module containing visualization code for creating topic distribution plots for university professors 
and bubble plot indicating which professors belong to which dominant topic.
When run as a module, this will load a json dataset, create the visualizations and store 
the resulting plots to disk.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly
import plotly.offline
import plotly.graph_objs as go
from plotly import tools
import colorlover as cl

sns.set()

def prof_topic_plots(filename):
    """Creates and saves topic distribution plots for each professor in .png format"""
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
        image_location = f'plots/prof_topic_plots/tp_{row["prof_id"]}'
        image_locations.append(image_location)
        plt.savefig(image_location)
        plt.close()
    
    df['topic_plot_locations'] = image_locations

    return df


def process_df(input_df, topic_descriptions, num_topics=12):
    """ Prepare the topic dataframe for creating detailed topic plot."""
    # Step 1: Extract the dominant topic for each professor
    df = input_df.copy()
    df["Dominant_Topic"] = df["Dominant_Topics"].apply(lambda x: x[0])

    # Step 2: Get professors, average h-index and leading university by h-index for each topic.
    prof_list = []
    topic_avg_h_index = []
    topic_univ = []

    for topic_num in range(1, num_topics + 1):
        prof_list.append(df[df["Dominant_Topic"] == topic_num]['faculty_name'].values)
        topic_avg_h_index.append(df[df["Dominant_Topic"] == topic_num]['h_index'].mean())
        topic_univ_avg_h_index = df[df["Dominant_Topic"] == topic_num][['university_name','h_index']].groupby('university_name').mean()
        topic_univ.append(topic_univ_avg_h_index.idxmax()[-1])
        
    topic_df = pd.DataFrame({'dominant_topic_num':range(1, num_topics + 1),
                        'faculty_names': prof_list,
                        'avg_h_index': topic_avg_h_index,
                        'topic_univ': topic_univ})

    # Step 3: Add topic size as the number of professors in that topic
    topic_df["topic_size"] = topic_df["faculty_names"].apply(lambda x: len(x))

    # Step 4: Add professors names to be shown in while hovering over the detailed topic plot.
    hover_text = []

    for _, row in topic_df.iterrows():
        hover_text.append(', '.join(name) for name in row['faculty_names'])
        
    topic_df['text'] = topic_df['faculty_names'].apply(lambda x:'<br>'.join(x))
    
    # Step 5: Add topic descriptions
    topic_df["topic_description"] = topic_descriptions

    return topic_df

def detailed_topic_plot(topic_df):
    """Create an interactive bubble plot using plotly."""
    sizeref = 2. * max(topic_df['topic_size'])/(75**2)

    trace0 = go.Scatter(
        x=topic_df['dominant_topic_num'][topic_df['topic_univ'] == 'University of Texas--Austin (Cockrell)'],
        y=topic_df['avg_h_index'][topic_df['topic_univ'] == 'University of Texas--Austin (Cockrell)'],
        mode='markers',
        name='University of Texas at Austin',
        text=topic_df['text'][topic_df['topic_univ'] == 'University of Texas--Austin (Cockrell)'],
        marker=dict(
            symbol='circle',
            sizemode='area',
            color = 'rgb(204,85,0)',
            sizeref=sizeref,
            size=topic_df['topic_size'][topic_df['topic_univ'] == 'University of Texas--Austin (Cockrell)'],
            line=dict(
                width=2
            ),
        )
    )

    trace1 = go.Scatter(
        x=topic_df['dominant_topic_num'][topic_df['topic_univ'] == 'Stanford University'],
        y=topic_df['avg_h_index'][topic_df['topic_univ'] == 'Stanford University'],
        mode='markers',
        name='Stanford University',
        text=topic_df['text'][topic_df['topic_univ'] == 'Stanford University'],
        marker=dict(
            symbol='circle',
            sizemode='area',
            color = 'rgb(0, 100, 0)',
            sizeref=sizeref,
            size=topic_df['topic_size'][topic_df['topic_univ'] == 'Stanford University'],
            line=dict(
                color = 'rgb(196,30,58)',
                width=2
            ),
        )
    )

    trace2 = go.Scatter(
        x=topic_df['dominant_topic_num'][topic_df['topic_univ'] == 'Texas A&M University--College Station'],
        y=topic_df['avg_h_index'][topic_df['topic_univ'] == 'Texas A&M University--College Station'],
        mode='markers',
        name='Texas A&M University',
        text=topic_df['text'][topic_df['topic_univ'] == 'Texas A&M University--College Station'],
        marker=dict(
            symbol='circle',
            sizemode='area',
            color = 'rgb(128, 0, 0)',
            sizeref=sizeref,
            size=topic_df['topic_size'][topic_df['topic_univ'] == 'Texas A&M University--College Station'],
            line=dict(
                color = 'rgb(255, 255, 255)',
                width=2
            ),
        )
    )

    trace3 = go.Scatter(
        x=topic_df['dominant_topic_num'][topic_df['topic_univ'] == 'University of Tulsa'],
        y=topic_df['avg_h_index'][topic_df['topic_univ'] == 'University of Tulsa'],
        mode='markers',
        name='University of Tulsa',
        text=topic_df['text'][topic_df['topic_univ'] == 'University of Tulsa'],
        marker=dict(
            symbol='circle',
            sizemode='area',
            color = 'rgb(65,105,225)',
            sizeref=sizeref,
            size=topic_df['topic_size'][topic_df['topic_univ'] == 'University of Tulsa'],
            line=dict(
                color = 'rgb(255, 255, 255)',
                width=2
            ),
        )
    )


    data = [trace0, trace1, trace2, trace3]

    layout = go.Layout(
        title='Topic Average h-index v. Topic Number',
        xaxis=dict(
            title='Topic Number',
            gridcolor='rgb(255, 255, 255)',
            range=[0, 13],
            zerolinewidth=1,
            ticklen=5,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Topic Average h-index',
            gridcolor='rgb(255, 255, 255)',
            range=[13, 45],
            zerolinewidth=1,
            ticklen=5,
            gridwidth=2,
        ),
        
        legend=dict(
            font=dict(
                family='sans-serif',
                size=18,
                color='#000'
            ),
            bgcolor='#E2E2E2',
            bordercolor='#FFFFFF',
            borderwidth=2
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',      
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename = 'templates/detailed_topic_plot.html', auto_open=False)

def create_topic_description_table(topic_df):
    """Create the topic descriptions table."""
    bupu = cl.scales['9']['div']['RdYlGn']
    colors = cl.interp( bupu, 16 ) # Map color scale to 16 bins
    # cl.to_html(colors) # html for the colorscale

    trace0 = go.Table(
    columnorder = [1,2],
    columnwidth = [80,300],
    header = dict(
        values = ["<b>TOPIC NUMBER</b>", "<b>TOPIC DESCRIPTION</b>"],
        line = dict(color = 'white'),
        fill = dict(color = 'white'),
        align = ['center'],
        font = dict(color = 'black', size = 14),
        height = 80
        
    ),
    cells = dict(
        values = [topic_df.dominant_topic_num, topic_df.topic_description],
        line = dict(color = [np.array(colors)[topic_df.topic_size],np.array(colors)[topic_df.topic_size]]),
        fill = dict(color = [np.array(colors)[topic_df.topic_size],np.array(colors)[topic_df.topic_size]]),
        align = ['center', 'left'],
        font = dict(color = 'black', size = 12)
        ))

    data = [trace0]
    plotly.offline.plot(data, filename = 'templates/topic_descriptions.html', auto_open=False)
    

if __name__ == '__main__':
    # Create prof topic plots
    df_updated = prof_topic_plots('../data/json/final_gensim_database_LDAMallet.json')
    df_updated.to_json(path_or_buf='../data/json/final_gensim_database_LDAMallet.json')

    # Create topic_df
    topic_descriptions = ['Health, Safety and Environment: Design, emissions and risk optimization',
                     'Unconventional Reservoirs: Study geology, estimate reserves, forecast production and uncertainity analysis for shale plays',
                     'Chemical EOR: Experimental and field applications involving surfactants and polymer',
                     'Hydraulic Fracturing: Simulations for modeling fracture propagation, network, interaction and stresses',
                     'Production Engineering(Theoretical): Flow simulation models based on experimental work',
                     'Reservoir Simulation: Computational and numerical modeling',
                     'Thermal and Solvent EOR: Application of high temperature and solvents for reservoirs with heavy oil and asphaltene',
                     'Phase Behavior: Experimental and simulation work involving steam and solvent processes',
                     'Production Engineering(Field application): Pressure and Rate Transient Analysis',
                     'Drilling Engineering: Wellbore stability, stresses, cementing and loss of fluids',
                     'Petrophysics and Well Logging: Pore Scale Modeling, Porosity, NMR, resistivity and density logs',
                     'Formation Evaluation: Stimulation techniques including acidization, hydraulic fracturing to tackle formation damage'
                     ]
    topic_df = process_df(df_updated, topic_descriptions, num_topics=12)

    # Create detailed topic plot
    detailed_topic_plot(topic_df)

    # Create topic descriptions table
    create_topic_description_table(topic_df)