
# coding: utf-8

# # TEXT MINING

# In[58]:


import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K

import seaborn as sns
import pydotplus
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import nltk
nltk.download('stopwords')


# # Preprocessing

# In[59]:


#preprocessing
def review_to_wordlist( review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()

    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b=[]
    stemmer = english_stemmer #PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 5. Return a list of words
    return(b)


# In[60]:


#import data
filepath =  'amazon_test.csv'
data = pd.read_csv(filepath)
data


# In[61]:


data = data[data['review'].isnull()==False]


# # Train/Test Split
# 

# In[62]:


#split the data into training and testing 
train, test = train_test_split(data, test_size = 0.3)


# In[63]:


#Labels Exploration
sns.countplot(data['rating'])


# In[64]:


review_sentiment = pd.Series(train['review']).astype(str)


# # Sentiment analyses

# In[65]:


#Applying sentiment analyses
#sentiment gives polarity(negative) and subjectivity(positive) values
import textblob
from textblob import TextBlob
review_sentiment[:50].apply(lambda x: TextBlob(x).sentiment)


# In[66]:


train['sentiment'] = review_sentiment.apply(lambda x: TextBlob(x).sentiment[1])
train[['review','sentiment']].head(30)


# In[68]:


#Apply Preprocessing
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append( " ".join(review_to_wordlist(review)))
    
clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append( " ".join(review_to_wordlist(review)))


# # MLP

# In[69]:


#MLP

#apply the MLP on the Tfidf Matrix

batch_size = 32
nb_classes = 5


# In[70]:


#TFidf transformation with ngrams
vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 1000, ngram_range = ( 1, 3 ),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(clean_train_reviews)
train_features = vectorizer.transform(clean_train_reviews)

test_features = vectorizer.transform(clean_test_reviews)


# In[71]:


X_train = train_features.toarray()
X_test = test_features.toarray()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

y_train = np.array(train['rating']-1)
y_test = np.array(test['rating']-1)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[72]:


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.mean(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]


# In[73]:


# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# In[74]:


# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


print("Training...")
model.fit(X_train, Y_train, epochs=5, batch_size=16, validation_split=0.1)

print("Generating test predictions...")
preds_mlp = model.predict_classes(X_test, verbose=0)


# In[75]:


print('prediction 1 accuracy: ', accuracy_score(test['rating'], preds_mlp+1))


# # CONFUSION MATRIX FOR MLP

# In[76]:


#confusion matrix for MLP
mlp_cfm = confusion_matrix(y_test,preds_mlp)
print("Confusion matrix:")
print(mlp_cfm, end='\n\n')
print('-'*25)
print(np.array([['TN', 'FP'],[ 'FN' , 'TP']]))


# In[77]:


df_heat_map = pd.DataFrame(mlp_cfm)


# In[78]:



fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_heat_map.corr(), vmax = 1,square=True)


# # LSTM

# In[79]:


max_features = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
maxlen = 80
batch_size = 32
nb_classes = 5


# In[80]:


# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['review'])
sequences_train = tokenizer.texts_to_sequences(train['review'])
sequences_test = tokenizer.texts_to_sequences(test['review'])


# In[81]:


print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In[82]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=1,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds_lstm = model.predict_classes(X_test, verbose=0)


# In[83]:


print('prediction 2 accuracy: ', accuracy_score(test['rating'], preds_lstm+1))


# # CONFUSION MATRIX FOR LSTM

# In[84]:


#confusion matrix for lstm
lstm_cfm = confusion_matrix(y_test,preds_lstm)
print("Confusion matrix:")
print(lstm_cfm, end='\n\n')
print('-'*25)
print(np.array([['TN', 'FP'],[ 'FN' , 'TP']]))


# In[85]:


df_heatmap_lstm = pd.DataFrame(lstm_cfm)


# In[86]:


#ploting for confusion matrix
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_heatmap_lstm.corr(), vmax = 1,square=True)


# # CNN

# In[87]:


nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2


# In[88]:


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

def max_1d(X):
    return K.max(X, axis=1)

model.add(Lambda(max_1d, output_shape=(nb_filter,)))
model.add(Dense(hidden_dims)) 
model.add(Dropout(0.2)) 
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[89]:


print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=6,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds_cnn = model.predict_classes(X_test, verbose=0)


# In[90]:


print('prediction 3 accuracy: ', accuracy_score(test['rating'], preds_cnn+1))


# # CONFUSION MATRIX FOR CNN

# In[91]:


#confustion matrix for CNN
cnn_cfm = confusion_matrix(y_test,preds_cnn)
print("Confusion matrix:")
print(cnn_cfm, end='\n\n')
print('-'*15)
print(np.array([['TN', 'FP'],[ 'FN' , 'TP']]))


# In[92]:


df_heatmap_cnn = pd.DataFrame(cnn_cfm)


# In[93]:


#ploting for confusion matrix
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df_heatmap_cnn.corr(), vmax = 1,square=True)

