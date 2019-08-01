# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 01:26:07 2019

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
#   posts = re.sub('[IiEe][NnSs][TtFf][JjPp]', '', posts ) # Removing type labels from posts if any
    posts = re.sub('\s+', ' ', posts)   # Removing extra whitespaces
    
    return posts

dataset['cleaned_posts'] = dataset['posts'].apply(extract_comments)



print(dataset['posts'][0])
print(dataset['cleaned_posts'][0])

count = 0
def make_corpus(posts):

    posts = re.sub('[^a-zA-Z]', ' ', posts)
    posts = posts.lower()
    posts = posts.split()
    
    ps = PorterStemmer()
    posts = [ps.stem(word) for word in posts if not word in set(stopwords.words('english'))]
        
    posts = ' '.join(posts)
    global count
    count += 1
    print(count)
    return posts

# =============================================================================
# # RUN ONLY ONE TIME TO MAKE CORPUS FILE (takes approx. 2 hours)
# corpus = list(dataset['cleaned_posts'].apply(make_corpus))
# 
# # Dumping corpus file
# import pickle
# outfile = open('old_corpus_file', 'wb')
# pickle.dump(corpus, outfile)
# =============================================================================

# Calculating average posts to set max_features
posts_len = dataset['cleaned_posts'].apply(len)
avg_posts_len = int(posts_len.mean())

# Loading corpus_file
infile = open('old_corpus_file', 'rb')
corpus = pickle.load(infile)
infile.close()


# Converting textual data into numerical format
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = avg_posts_len)
features = cv.fit_transform(corpus).toarray()

with open("vectorize.pkl","wb") as outfile:
    pickle.dump(cv, outfile)

labels = dataset.iloc[:, 0].values
# =============================================================================
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# labels = labelencoder.fit_transform(labels)
# =============================================================================

# =============================================================================
# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder()
# labels = onehotencoder.fit_transform(labels.reshape(-1,1)).toarray()
# =============================================================================

# Handling imbalancing of data using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
features, labels = sm.fit_sample(features, labels)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)


# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit( features_train, labels_train )
# labels_pred = gnb.predict( features_test )
# =============================================================================


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score
print( accuracy_score(labels_test, labels_pred) )

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = features_train, y = labels_train, cv = 10)
print ("mean accuracy is",accuracies.mean())
print (accuracies.std())


# =============================================================================
# comment = r"I do not need any friend.I failed a public speaking class a few years ago and I've sort of learned what I could do better were I to be in that position again. A big part of my failure was just overloading myself with too... I like this person's mentality. He's a confirmed INTJ by the way. Move to the Denver area and start a new life for myself."
# comment = make_corpus(comment)
# comment = cv.transform(list([comment])).toarray()
# 
# prediction = classifier.predict(comment)
# # prediction = labelencoder.transform(prediction)
# 
# print("Your Type :",prediction[0])
# 
# mbti_type = [personality_types[char] for char in str(prediction[0])]
# mbti_type = ' - '.join(mbti_type)
# print('('+mbti_type+')')
# =============================================================================


with open('model.pkl', 'wb') as outfile:
    pickle.dump(classifier, outfile, pickle.HIGHEST_PROTOCOL)

