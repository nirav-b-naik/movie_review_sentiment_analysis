#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:50:32 2023

@author: nirav-ubuntu
"""

"""
### Import Module
"""

# To remove punctuation
import string

# Array operation
import numpy as np

# Regex Operation
import re

import os

# Stopword remove
import nltk
from nltk.corpus import stopwords

# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer

# Deep Neural Network
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Save Model in Pickel Format
import pickle

"""
### UDF for Preprcessing and Cleaning
"""

# function to read contents from text file
def load_doc(filename):
    
    # open file read only
    file = open(filename,'r')
    
    # read all text
    text = file.read()
    
    # close file
    file.close()
    
    # return text data
    return text

# function to load file vocab file loaded
def vocab_set():
    
    # path of vocab file
    vocab_filename = './movie_review/vocab.txt'
    
    # reading vocab file
    vocab = load_doc(vocab_filename)
    
    # converting to set
    vocab = set(vocab.split())
    
    return vocab

# turn a doc into clean tokens
def clean_doc(doc):
    
    # split into tokens by white space
    tokens = doc.split()
    
    # prepare regex for char filtering
    re_punc = re.compile("[%s]" % re.escape(string.punctuation))
    
    # remove punctuation from each word
    tokens = [re_punc.sub( "", w) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    
    # filter out stop words
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if not w in stop_words]
    
    # filter out short tokens
    tokens = [word for word in tokens if len(word)>1]
    
    return tokens

# Load docs, clean and return line of tokens
def doc_to_line(filename):
    
    # load the doc
    doc = load_doc(filename)
    
    # clean doc
    tokens = clean_doc(doc)
    
    # loading vocab set
    vocab = vocab_set()
    
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    
    return " ".join(tokens)

# load all docs from dictionary
def process_docs(directory):
    
    lines = list()
    
    # walk through all files and folders
    for filename in os.listdir(directory):
        
        # create the full path
        path = directory + "/" + filename
        
        # load and clean data
        line = doc_to_line(path)
        
        # add to list
        lines.append(line)
    
    return lines

# load all docs from dictionary
def process_docs(directory):
    
    lines = list()
    
    # walk through all files and folders
    for filename in os.listdir(directory):
        
        # create the full path
        path = directory + "/" + filename
        
        # load and clean data
        line = doc_to_line(path)
        
        # add to list
        lines.append(line)
    
    return lines

# function to load and clean entire dataset
def load_clean_dataset():
    
    # load documents
    neg = process_docs('./movie_review/neg')
    pos = process_docs('./movie_review/pos')
    
    docs = neg + pos
    
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    
    return docs,labels

# function to fit a tokenizer
def create_tokenizer(lines):
    
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(lines)
    
    return tokenizer

# function to define and create model
def define_model(n_words):
    
    # define network
    model = Sequential()
    
    # Dense Layer 1
    model.add(Dense(50,
                   input_shape = (n_words,),
                   activation = 'relu'))
    
    # Dense Layer 2 (output layer)
    model.add(Dense(1,
                    activation = 'sigmoid'))
    
    # compilation
    model.compile(loss = 'binary_crossentropy',
                 optimizer = 'adam',
                 metrics = ['accuracy'])
    
    # summarize defined model
    model.summary()
    
    # plot model
    plot_model(model,
              to_file = 'model.png',
              show_shapes = True)
    
    return model

# classify the review as negative or positive
def predict_sentiment(review,model,tokenizer):
    
    # clean
    tokens = clean_doc(review)
    
    # loading vocab set
    vocab = vocab_set()
    
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    
    # convert to line
    line = ' '.join(tokens)
    
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode = 'binary')
    
    # predict sentiments
    yhat = model.predict(encoded,verbose = 0)
    
    # retrieve predicted percentege and label
    percent_pos = yhat[0,0]
    
    if round(percent_pos)==0:
        
        return (1-percent_pos), 'NEGATIVE'
    
    return percent_pos, 'POSITIVE'

if __name__=="__main__":
    
    # load all reviews
    # train dataset
    train_docs , ytrain = load_clean_dataset()
    
    # test dataset
    test_docs , ytest = load_clean_dataset()
    
    # create the tokenizer
    # this is object
    tokenizer = create_tokenizer(train_docs)
    
    # encode data
    # convert text doc to binary matrix
    #(i.e. if word present 1 else 0)
    X_train = tokenizer.texts_to_matrix(train_docs, mode = 'binary')
    X_test = tokenizer.texts_to_matrix(test_docs, mode = 'binary')
    
    # Define the network
    n_words = X_train.shape[1]
    model = define_model(n_words)
    
    # fit the network
    model.fit(X_train,np.array(ytrain),
              validation_data=[X_test,np.array(ytest)],
             epochs=10,
             batch_size = 10)
    
    # Save the model using pickle module
    pickle.dump(model,open('sentiment_model.pkl','wb'))
    
    # Save the tokenizer using pickle module
    pickle.dump(tokenizer,open('sentiment_token.pkl','wb'))
    
    text = 'best movie ever!! it was great. i recommend it.'

    percent , sentiment = predict_sentiment(text,model,tokenizer)

    print(f"Review: {text}\nSentiment: {sentiment} ({round(percent*100,2)}%)")
