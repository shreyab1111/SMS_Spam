from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from matplotlib import pyplot as plt

df=pd.read_csv('spam.csv', encoding='ISO-8859-1')

df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df2=df.rename(columns={'v1':'label','v2':'data'})    

df2['b_labels']=df2['label'].map({'ham':0,'spam':1})

Y=df2['b_labels'].values

count_vectorizer=CountVectorizer(decode_error='ignore')
X=count_vectorizer.fit_transform(df2['data'])

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.35)

model=MultinomialNB()
model.fit(Xtrain,Ytrain)
print("Train score: ",model.score(Xtrain,Ytrain))
print("Test score: ",model.score(Xtest,Ytest))

def visualize(label):
    word=''
    for m in df2[df2['label']==label]['data']:
        m=m.lower()
        word = word+ m + ' '
    wordcloud=WordCloud(width=600, height=400).generate(word)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
visualize('spam')
visualize('ham')

df2['predictions']=model.predict(X)

#things that should be spam but are not correctly predicted
sneaky_spam=df2[(df2['predictions']==0) & (df2['b_labels']==1)]['data']
for m in sneaky_spam:
    print(m)
    
#things that should not be spam but are incorrectly predicted as spam
not_actual_spam=df2[(df2['predictions']==1) & (df2['b_labels']==0)]['data']
for m in not_actual_spam:
    print(m)    
