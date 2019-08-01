# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 02:09:34 2019

@author: Himanshu Rathore
"""

#from model import make_corpus
import pickle
    # de-serialize mlp_nn.pkl file into an object called model using pickle
with open('model.pkl', 'rb') as infile:
    classifier = pickle.load(infile)
    
with open('vectorize.pkl', 'rb') as file:
    cv = pickle.load(file)
 
def prediction(user_data):

        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        
        global classifier
        
        user_data = list(map(lambda x: x.lower(), user_data))
        
        ps = PorterStemmer()
        user_data = [ps.stem(word) for word in user_data if not word in set(stopwords.words('english'))]
            
        user_data = ' '.join(user_data)
        
        ptype = classifier.predict(cv.transform(list([user_data])).toarray())
        personality_types = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
        mbti_type = [personality_types[char] for char in str(ptype[0])]
        mbti_type = ' - '.join(mbti_type)
        
        return mbti_type