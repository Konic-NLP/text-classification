# -*- coding: utf-8 -*-
"""Untitled0.ipynb


"""

!git clone https://github.com/Perez-AlmendrosC/dontpatronizeme.git


from dont_patronize_me import DontPatronizeMe
dpm = DontPatronizeMe('/content/drive/MyDrive', 'dontpatronizeme_pcl.tsv')

dpm.load_task1()
data=dpm.train_task1_df

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
import numpy as np



tfidf=TfidfVectorizer(min_df=3,max_features=10000,max_df=100)




data=pd.read_csv('/content/drive/MyDrive/train.csv')
test=pd.read_csv('/content/drive/MyDrive/test.csv')

pcldf=data[data.label==1]
npos=len(pcldf)
train=pd.concat([pcldf,data[data.label==0][:npos*2]])

labels=train['label']




feature=tfidf.fit_transform(train['text'].astype(str))
testfeature=tfidf.transform((testX.values).astype(str))


SVM=SVC(C=10,kernel='linear',max_iter=30)
SVM.fit(features[1:],labels)

testX=test['text']

y_predict=SVM.predict(test)

y_pred=SVM.predict(testfeatures[1:])

f1_score(test['label'], y_pred)


data=dpm.train_task1_df
data.dropna(how='any')
feature=tfidf.fit_transform(data['text'])

import numpy as np

data.replace(to_replace=pd.NA, value=0, inplace=True)

import numpy as np
labels=np.array(labels).reshape(-1,1)


'''
for word embedding method
'''


def get_embed(file):
  embed={}
  with open(file)as f:
    for i in f:
      i=i.split()
      embed[i[0]]=i[1:]
  return embed

embed=get_embed('/content/drive/MyDrive/glove.6B.300d.txt')


def get_vec(text,embed=embed):
 
  vec=np.zeros(300)
  vec.astype('float64')
  k=1
 
  num=data['text'][data['text'].values==text].index
  print(num)
  for i in text.split():
    if i not in embed.keys():
      continue
    # print(np.array(embed[i]).shape)
    if i not in tfidf.vocabulary_.keys():
      weight=0
    else:
      weight=feature.todense()[num,tfidf.vocabulary_[i]]
      np.array(embed[i]).astype('float64')
      value=weight[0][0].reshape(1,)*embed[i]
   

    
    k+=1
  return vec


def get_features

features=data['text'].apply(lambda x: get_vec(x,embed))

features=np.zeros(300).astype('float64')
for i in feature:
  features=np.row_stack((features,i))


testfeature=test['text'].astype(str).apply(lambda x: get_vec(x,embed))

testfeatures=np.zeros(300).astype('float64')
for i in testfeature:
  testfeatures=np.row_stack((testfeatures,i))

# it's the same as above, convert (len,)(dim,) into (len,dim)
features=[]
for i in feature:
  features.append(i)




testfeatures=[]
for i in testfeature:
  testfeatures.append(i)


y_pred=SVM.predict(testfeatures)
f1=f1_score(test['label'],y_pred)
print(f1)

f1_score(Y_test, y_pred)

