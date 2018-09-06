# professor-profiler
The profiler tool gives a holistic profile of a graduate school professor including his personal information, tags based on the research areas being worked on by him/her and ratings/reviews given by past students in the classes taught by him/her. This project is geared towards assisting prospective graduate students in choosing which graduate schools to apply to or professors they would like to do research collaboration with after getting admitted to a particular school. On a more broad level, the tool helps to locate and track the work being done by research experts in a particular specialty (for example: hydraulic fracturing).

Currently, the scope of this project is limited to one major: petroleum engineering and covers research professors in the top-4 US petroleum schools. Only publications published in OnePetro and ScienceDirect since 2010 have been used. A total of 88 professors and ~5700 publications have been included in the dataset.

## Data Acquisition
All of the data used for building this tool was web scraped from the sources given below:

* University rankings and information: https://www.usnews.com/ 
* Faculty information: 
    * University of Texas: https://www.pge.utexas.edu/
    * Stanford University: https://pangea.stanford.edu/ere/
    * Texas A&M University: https://engineering.tamu.edu/petroleum/index.html
    * University of Tulsa: https://engineering.utulsa.edu/petroleum-engineering/
* Faculty ratings: http://www.ratemyprofessors.com/
* Publication and Faculty research profile information: https://scholar.google.com/, https://www.scopus.com/
* Publication abstracts and titles: https://www.onepetro.org/, https://www.sciencedirect.com/

## Modeling Process
There were four steps involved in the modeling process: Data cleaning and processing, clustering, topic modeling and model comparison and evaluation.

**Stage 1: Data cleaning and processing**
A typical natural language processing pipeline was set up using NLTK. Domain-specific stopwords were added such as oil, gas, water etc. since they were expected to be in high frequency in all publications.

**Stage 2: Clustering**
K-Means clustering model was built using scikit-klearn which would cluster professors by research areas. Visualization of the clusters was created using TruncatedSVD and the number of clusters were optimized using the elbow method and silhouette analysis. The problem with this model was that one professor could only work in one area i.e. it was a hard clustering model. 

**Stage 3: Topic Modeling**
Topic modeling allows professors to have multiple research areas and each word to be present in multiple research areas with different probabilities. These probabilistic models were built using the gensim and scikit-learn libraries, which utilize latent dirichlet allocation and non-negative matrix factorization algorithms. Interactive visualization of the topics was generated with pyLDAvis and the number of topics was optimized using coherence plots. 

**Stage 4: Modeling Comparison and Evaluation**
Four metrics were used to compare and evaluate the topic models namely visualization of clusters/topics, coherence score, domain knowledge i.e. petroleum engineering and run time of algorithms. LDA Mallet, which is a Java based topic model, gave the best results. Gensim provides a Python wrapper around the Java version and that was utilized for the modeling in this work.

For more information about LDA Mallet, please see here: http://mallet.cs.umass.edu/

**Assumptions**
* Text used in the model was only taken from paper titles, abstracts and the research description of the professor. It was assumed that most of the critical information about a publication would be contained in the paper title and abstract.
* The search algorithm has a weighted criteria based on text-similarity with the user input, h-index of the professor, number of papers of a professor in the database used in the model and the university rank.

## Run the recommender
To run the recommender with your choice of model (NMF, LSI, LDA, LDAMallet(default), KMeans), first install the dependencies mentioned in requirements.txt and then:

```
python app.py 
```

## Website demo
You can watch the demo of the website on the link given below:
https://drive.google.com/file/d/1Gvor95sAReVSRml-8bBJkZv2Jb8IjheK/view?usp=sharing

## Future Work
The next steps would involve scaling the project to include more majors, schools and journals and carry out a graph and time-series analysis of the research topics.
