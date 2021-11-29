import numpy as np
import random
import word2vec
import jieba
import pandas as pd
from keras import Sequential
from keras.layers import LSTM,Bidirectional,Activation,Dense,Flatten
from keras_preprocessing import sequence
from tensorflow.python.client import device_lib
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    path = '/home/linke3090/Downloads/'

    w2v = word2vec.load(path + "sgns.weibo.word.txt")
    pd_all = pd.read_csv(path + 'weibo_senti_100k.csv')  # 两列:label   review

    print(pd_all.head())
    print('评论数目（总体）：%d' % pd_all.shape[0])
    # print('评论数目（正向）：%d' % pd_all[pd_all.label == 1].shape[0])
    # print('评论数目（负向）：%d' % pd_all[pd_all.label == 0].shape[0])
    positive = pd_all[pd_all.label==1].values.tolist()
    negetive = pd_all[pd_all.label==0].values.tolist()
    data_all = pd_all.values.tolist()

    strs = []
    strs_label = []

    for i, str in data_all:
        strs.append(str)
        strs_label.append(i)

    shuffle_index = np.random.permutation(np.arange(len(strs)))
    strs = np.array(strs)[shuffle_index]
    y = np.array(strs_label)[shuffle_index]

    x = np.zeros(shape=(len(strs),128,300), dtype=np.float32)

    for i, str in enumerate(strs):
        res_cuts = jieba.cut(str[:128])
        for j,res_cut in enumerate(res_cuts):
            if res_cut in w2v:
                x[i,j,:] = w2v[res_cut]

    model = Sequential()
    model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True),input_shape=(128,300)))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    print(model.summary())
    print('gpu is available? ', tf.test.is_built_with_cuda())

    history = model.fit(x, y, validation_split=0.1, batch_size=256, epochs=5)
    model.save('bilstm.h5')

    np.save('history.npy', history.history)

