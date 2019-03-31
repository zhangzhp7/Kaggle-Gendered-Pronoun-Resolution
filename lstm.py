#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:46:50 2019

@author: zhangzhaopeng
"""

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,GRU, SpatialDropout1D, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional

import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/kaggle/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train_processed, x_valid_processed, y_train, y_valid, word_index, embedding_matrix = pickle.load(fp)
fp.close()

data_path = '/Users/zhangzhaopeng/统计学习/kaggle/cleaned_text.pkl'
fp = open(data_path, 'rb')
x_train, x_valid = pickle.load(fp)
fp.close()

EMBEDDING_DIM = 50
maxLen = len(max(x_train, key=len))
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxLen,
                            trainable=True
                            )
input_layer = Input(shape=(maxLen,), name = 'input')
X = embedding_layer(input_layer)
X = SpatialDropout1D(rate=0.5)(X)
X = Bidirectional(LSTM(128, return_sequences = True), name='bidirectional_lstm')(X)
X = GlobalMaxPool1D(name='global_max_pooling1d')(X) 

X = Dense(units=32, activation='relu', name='dense_1')(X)
X = Dropout(0.65)(X)
 
output_layer = Dense(units = 3, activation = 'softmax', name = 'output')(X)
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(x_train_processed, y_train, epochs = 10, batch_size = 256, shuffle = True, validation_data=(x_valid_processed, y_valid))






