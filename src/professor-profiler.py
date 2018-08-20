from cleaning import database_cleaner

import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy  import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

tamu_df = database_cleaner('../data/tamu_database.json')
# For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
df = tamu_df[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
missing = df['paper_titles'] == ''
print(f'{missing} faculties have missing papers in tamu_database')
print('Running nlp-pipeline on faculties with non-missing papers...')
