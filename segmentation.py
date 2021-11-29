import torch
from torch import nn
import random
import math
import pandas as pd
import os
import jieba

path = '~/Downloads/'
pd_all = pd.read_csv(path + 'weibo_senti_100k.csv')   # 两列:label   review
print(pd_all.head())
print('评论数目（总体）：%d' % pd_all.shape[0])
print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])

labels = pd_all["label"].values.tolist()
reviews = pd_all["review"].values.tolist()
seg_list = jieba.cut(reviews[0], cut_all='True')
print('/'.join(seg_list))

dict_set = set()

for content in reviews:
    seg_list = jieba.cut(content, cut_all='True')
    for s in seg_list:
        dict_set.add(s)

dict_set.add("<unk>")
dict_list = []
i=0
for s in dict_set:
    dict_list.append([s, i])
    i+=1

dict_txt = dict(dict_list)
with open("dict.txt", "w", encoding='utf=8') as f:
    f.write(str(dict_txt))

