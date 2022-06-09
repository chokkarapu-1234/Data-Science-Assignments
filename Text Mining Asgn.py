# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:18:53 2022

@author: varun
"""
 
import pandas as pd
import re
from textblob import TextBlob
from textblob import TextBlob
df = pd.read_csv("C:\\Users\\varun\\Documents\\excelr assignments\\Text Mining\\Elon_musk.csv",encoding = 'Latin1')
df
pd.set_option('display.max_colwidth', -1)
df.shape
list(df)
df.head
list(df)
df.isnull().sum()
df

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

#Defining a dcitionary containing all the emojis and their meanings
emojis={':)':'smile',':-)':'smile',';d':'wink',':-E':'vampire',':(':'sad',
        ':-(':'sad',':-<':'sad',':P':'raspberry',':O':'surprised',
        ':-@':'shocked',':@':'shocked',':-$':'confused',':\\':'annoyed',
        ':#':'mute',':X':'mute',':^)':'smile',':-&':'confused','$_$':'greedy',
        '@@':'eyeroll',':-!':'confused',':-D':'smile',':-0':'yell','O.o':'confused',
        '<(-_-)>':'robot','d[-_-]b':'dj',":'-)":'sadsmile',';)':'wink',
        ';-)':'wink','O:-)':'angel','O*-)':'angel','(:-D':'gossip','=^.^=':'cat'}


#Defining a function to clean the data
def clean_text(kit):
    kit=str(kit).lower()
    kit=re.sub(r"@\S+",r'',kit)
    
    for i in emojis.keys():
        kit=kit.replace(i,emojis[i])
    kit=re.sub("\s+",' ',kit)
    kit=re.sub("\n",' ',kit)
    letters=re.sub('[^a-zA-Z]',' ',kit)
    return letters

#Defining a function to remove the stop words        
def stops_words(words):
    filter_words=[]
    for w in words:
        if w not in stop_words:
            filter_words.append(w)
    return filter_words


#Defining a function for sentiment analysis
def getSubjectivity(tex):
    return TextBlob(tex).sentiment.subjectivity

def getPolarity(tex):
    return TextBlob(tex).sentiment.polarity

def getAnalysis(score):
    if int(score)<0:
        return 'Negative'
    elif int(score)==0:
        return 'Neutral'
    elif int(score)>0:
        return 'Positive'

df['Text']=df['Text'].apply(lambda x:clean_text(x))

#Removing stop words
df['Text']=df['Text'].apply(lambda x:x.split(" "))
df['Text']=df['Text'].apply(lambda x:stops_words(x))


#Stemming
from nltk.stem import PorterStemmer
stem=PorterStemmer()
df['Text']=df['Text'].apply(lambda x: [stem.stem(k) for k in x])

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
df['Text']=df['Text'].apply(lambda x: [lemm.lemmatize(j) for j in x])

df['Text']=df['Text'].apply(lambda x: ' '.join(x))

#Preparing a target variable which shows the sentiment i.e, Subjectivity and Polarity
df['sentiment_subj']=df['Text'].apply(lambda x:getSubjectivity(x))
df['sentiment_subj'].describe()    

df['sentiment_pol']=df['Text'].apply(lambda x:getPolarity(x))
df['sentiment_pol'].describe()

sentiment=[]
for i in range(0,1999,1):
    if df['sentiment_pol'].iloc[i,] < 0:
        sentiment.append('Negative')
    elif df['sentiment_pol'].iloc[i,] == 0:
        sentiment.append('Neutral')
    else:
        sentiment.append('Positive')
sentiment
Sentiment=pd.DataFrame(sentiment)
Sentiment.set_axis(['sentiment'],axis='columns',inplace=True)
df_new=pd.concat([df,Sentiment],axis=1)
df_new.shape
list(df_new)


import seaborn as sns
sns.distplot(df_new['sentiment_subj'])
sns.distplot(df_new['sentiment_pol'])
sns.countplot(df_new['sentiment'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
word_cloud = WordCloud().generate('df_new['sentiment']')
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()



#Splitting into train and test
from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.25,random_state=41)

df_train_clean=data_train['Text']
df_test_clean=data_test['Text']

#Vectorize the data
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(use_idf=True)
df_train_clean=vector.fit_transform(data_train_clean)
df_test_clean=vector.transform(data_test_clean)

df_train_clean.toarray()
df_train_clean.toarray().shape

vector.get_feature_names()



'''

from sklearn.metrics import accuracy_score,recall_score,precision_score
import pickle


def model_perf(model):
    Y_pred=model.predict(df_test_clean)
    acc=accuracy_score(df_test['sentiment'],Y_pred)
    rec=recall_score(df_test['sentiment'],Y_pred,pos_label='negative')
    prec=precision_score(df_test['sentiment'],Y_pred,pos_label='negative')
    
    return(acc,rec,prec)

###Apply any ML algorithm and check which gives out the best accuracy########

from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

BNB=BernoulliNB(alpha=2)
SVC=LinearSVC()
LR=LogisticRegression(C=2,max_iter=1000,n_jobs=-1)

models=[BNB,SVC,LR]
model_scores={}
model_fitted={}
for i in models:
    i.fit(,df_train['sentiment'])
    accur=model_perf(i)
    model_scores[i.__class__.__name__]=accur[0]
    model_fitted[i.__class__.__name__]=i
best_model=max(model_scores,key=model_scores.get)

#Saving the model
filename=best_model+.'.pickle'

with open(r'saved_model/'+filename,'wb') as new_model:
    pickle.dump(model_fitted[best_model],new_model)
    
with open('saved_model/tfvectorizer.pickle','wb') as file:
    pickle.dump(vector,file)
'''







X=df['Text']
list(X)
X.head()
X.shape
import nltk
import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#;|);|:;|!;|<;|>;|_;|*;|~;|-;'

def clean(zen):
    zen = re.sub(HANDLE, ' ', zen)
    zen = re.sub(LINK, ' ', zen)
    zen = re.sub(SPECIAL_CHARS, ' ', zen)
    return zen

for i in range(0,1999,1):
    data.iloc[i,1]=data.iloc[i,1].apply(clean)

X.iloc[5,]=X.iloc[5,].apply(clean)
