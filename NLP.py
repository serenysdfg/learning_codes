# -*- coding: utf-8 -*-
# @createTime : 2019.10.28 13:27
# @Author : sereny
#KERAS的中文+注释,分类

# ! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

maxlen = 100
config_path = '../model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../model/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

# 数据获取00.1
neg = pd.read_excel('../model/neg.xls', header=None)
pos = pd.read_excel('../model/pos.xls', header=None)

data = []

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


# seq_padding是用来处理输入的训练数据的，模型的输入必须是一个矩阵形式
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# 生成网络输入
class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

def build_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(1, activation='sigmoid')(x)
    # p = Dense(nclass, activation='softmax')(x) #多分类

    model = Model([x1_in, x2_in], p)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])
    print(model.summary())
    return model
model = build_bert()
# bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
# for l in bert_model.layers:
#     l.trainable = True
# x1_in = Input(shape=(None,))
# x2_in = Input(shape=(None,))
# x = bert_model([x1_in, x2_in])
# x = Lambda(lambda x: x[:, 0])(x)
# p = Dense(1, activation='sigmoid')(x)
# model = Model([x1_in, x2_in], p)
# model.compile(
#     loss='binary_crossentropy',  # 二分类
#     optimizer=Adam(1e-5),  # 用足够小的学习率
#     metrics=['accuracy']
# )
# model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)