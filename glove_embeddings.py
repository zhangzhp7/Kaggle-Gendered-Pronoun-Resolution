#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:37:31 2019

@author: zhangzhaopeng
"""

import os
import numpy as np
import pickle

## import data
data_path = '/Users/zhangzhaopeng/统计学习/kaggle/data_preprocessing.pkl'
fp = open(data_path, 'rb')
x_train_processed, x_valid_processed, y_train, y_valid, word_index = pickle.load(fp)
fp.close()

GLOVE_DIR = '/Users/zhangzhaopeng/统计学习/kaggle/embeddings'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 50
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
import pickle
data = (x_train_processed, x_valid_processed, y_train, y_valid, word_index, embedding_matrix)
fp = open('/Users/zhangzhaopeng/统计学习/kaggle/data_embeddings.pkl', 'wb')
pickle.dump(data, fp)
fp.close() 



