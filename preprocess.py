#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:41:04 2019

@author: zhangzhaopeng
"""

import pandas as pd

#test = pd.read_csv("../input/gendered-pronoun-resolution/test_stage_1.tsv",sep = '\t')
#print(test.head())
#sub = pd.read_csv("../input/gendered-pronoun-resolution/sample_submission_stage_1.csv",sep = '\t')
#print(sub.head())

#load data
gap_test = pd.read_csv("/Users/zhangzhaopeng/统计学习/kaggle/gap-coreference-master/gap-test.tsv",sep = '\t')
gap_valid = pd.read_csv('/Users/zhangzhaopeng/统计学习/kaggle/gap-coreference-master/gap-validation.tsv', sep = '\t')
train = pd.concat([gap_test, gap_valid], ignore_index=True, sort=False)
print(gap_test.shape, gap_valid.shape, train.shape)
test = pd.read_csv('/Users/zhangzhaopeng/统计学习/kaggle/gap-coreference-master/gap-development.tsv', sep = '\t')
print(test.shape)
data_all = pd.concat([train, test], ignore_index=True, sort=False)

# 分词
def doc2word(data):
    doc2words = []
    for line in data:
        line = line.split(' ')
        doc2words.append(line)
    return doc2words
doc2words = doc2word(data_all.Text)

#   去掉分词前后无用的字符
all_flags=[',', '.', '!', '?', ';', "''", "`",'(',')',':']
def text_clean(doc2words):
    text = []
    for line in doc2words:
        temp_line = []
        for word in line:
            for flag in all_flags:
                word = word.strip(flag)
            temp_line.append(word)
        text.append(temp_line)
    return text
text_preprocessed = text_clean(doc2words)

x_train = text_preprocessed[:train.shape[0]]
x_valid = text_preprocessed[train.shape[0]:]

# Tokenize the text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def build_word_tokenizer(text):
    vocabulary = []
    for sent in text:
        for word in sent:
            vocabulary.append(word)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([vocabulary])
    return tokenizer
tokenizer = build_word_tokenizer(x_train)
x_train_token = tokenizer.texts_to_sequences(x_train)
x_valid_token = tokenizer.texts_to_sequences(x_valid)

# pad the sentence
maxLen = len(max(x_train, key=len))
x_train_processed = pad_sequences(x_train_token, maxlen=maxLen)
x_valid_processed = pad_sequences(x_valid_token, maxlen=maxLen)

word_index = tokenizer.word_index

# process labels
def row_to_y(row):
    if row.loc['A-coref']:
        return 0
    if row.loc['B-coref']:
        return 1
    return 2

y_train = train.apply(row_to_y, axis = 1)
y_valid = test.apply(row_to_y, axis = 1)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

import pickle
data = (x_train_processed, x_valid_processed, y_train, y_valid, word_index)
fp = open('/Users/zhangzhaopeng/统计学习/kaggle/data_preprocessing.pkl', 'wb')
pickle.dump(data, fp)
fp.close()        

text = (x_train, x_valid)
fp = open('/Users/zhangzhaopeng/统计学习/kaggle/cleaned_text.pkl', 'wb')
pickle.dump(data, fp)
fp.close() 












