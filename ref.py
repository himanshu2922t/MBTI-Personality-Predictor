# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 02:07:00 2019

@author: Himanshu Rathore
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:03:57 2019

@author: Himanshu Rathore
"""

# importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# importing Dataset and changing first column into categorical datatype
dataset = pd.read_csv('mbti_dataset.csv')

# exploring dataset
print( dataset.shape )
print( dataset.head() )
print( dataset.info() )
print( dataset.describe() )

# check data types for each column
print (dataset.dtypes)

# Check if any NaN values in dataset
dataset.isnull().any(axis=0)

# Making Dictionary for types
personality_types = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}


def extract_comments(posts):
    posts = re.sub('\|\|\|', ' ', posts)    # Removing three pipeline characters
    posts = re.sub('https?\S+', '', posts)  # Removing links 
# =============================================================================
#     posts = re.sub('[IiEe][NnSs][TtFf][JjPp]', '', posts ) # Removing type labels from posts if any
# =============================================================================
    posts = re.sub('\s+', ' ', posts)   # Removing extra whitespaces
    
    return posts

dataset['cleaned_posts'] = dataset['posts'].apply(extract_comments)

map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
dataset['I-E'] = dataset['type'].astype(str).str[0]
dataset['I-E'] = dataset['I-E'].map(map1)
dataset['N-S'] = dataset['type'].astype(str).str[1]
dataset['N-S'] = dataset['N-S'].map(map2)
dataset['T-F'] = dataset['type'].astype(str).str[2]
dataset['T-F'] = dataset['T-F'].map(map3)
dataset['J-P'] = dataset['type'].astype(str).str[3]
dataset['J-P'] = dataset['J-P'].map(map4)

print(dataset['posts'][0])
print(dataset['cleaned_posts'][0])


# Loading corpus_file
infile = open('corpus_file', 'rb')
corpus = pickle.load(infile)
infile.close()

# Calculating average posts to set max_features
posts_len = dataset['cleaned_posts'].apply(len)
avg_posts_len = int(posts_len.mean())

# Converting textual data into numerical format
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = avg_posts_len)
features = cv.fit_transform(corpus).toarray()

labels = dataset.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(labels)


IE_labels = dataset.iloc[:,3].values
NS_labels = dataset.iloc[:,4].values
TF_labels = dataset.iloc[:,5].values
JP_labels = dataset.iloc[:,6].values

# Handling imbalancing of data using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
features, labels = sm.fit_sample(features, labels)
IE_features, IE_labels = sm.fit_sample(features, IE_labels)
NS_features, NS_labels = sm.fit_sample(features, NS_labels)
TF_features, TF_labels = sm.fit_sample(features, TF_labels)
JP_features, JP_labels = sm.fit_sample(features, JP_labels)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)
features_train, features_test, labels_train, labels_test = train_test_split(IE_features, IE_labels, test_size = 0.20, random_state = 0)
features_train, features_test, labels_train, labels_test = train_test_split(NS_features, NS_labels, test_size = 0.20, random_state = 0)
features_train, features_test, labels_train, labels_test = train_test_split(features, TF_labels, test_size = 0.20, random_state = 0)
features_train, features_test, labels_train, labels_test = train_test_split(features, JP_labels, test_size = 0.20, random_state = 0)

classifier = RandomForestClassifier(n_estimators=100)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)

print( accuracy_score(labels_test, labels_pred) )

from collections import Counter
Counter(labels)





