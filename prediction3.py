import numpy as np
import random
import word2vec
import jieba
import sys
from keras import Sequential, models
from keras.layers import LSTM, Bidirectional, Activation, Dense, Flatten
from keras_preprocessing import sequence

if __name__ == "__main__":

    if (len(sys.argv) >= 2):

        path = '/home/linke3090/Downloads/'

        w2v = word2vec.load(path + "sgns.weibo.word.txt")

        x = np.zeros(shape=(1, 128, 300), dtype=np.float32)
        model = models.load_model('bilstm.h5')
        # print(model.summary())
        for i, str in enumerate(sys.argv[1:]):
            res_cuts = jieba.cut(str)
            for j, res_cut in enumerate(res_cuts):
                if res_cut in w2v:
                    x[0, j, :] = w2v[res_cut]
                    # print(res_cut)
            res = model.predict(x)
            print(str)
            print(res)

    else:
        pass